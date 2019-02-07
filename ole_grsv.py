import torch


class _OLE(torch.autograd.Function):
    """
    this is based on the logic found in the original OLE repo:
    https://github.com/jlezama/OrthogonalLowrankEmbedding/blob/master/pytorch_OLE/OLE.py
    """

    @staticmethod
    def forward(ctx, z_geometry, z_geometry_svd, delta=1.0, eigThresh=1e-6):
        """
        """
        # gradients initialization
        obj_c = 0
        d_z_geometry = torch.zeros_like(z_geometry)
        N = z_geometry.size()[0]

        for svd in z_geometry_svd:
            v, s, u, class_indices = svd

            nuclear = torch.sum(s)

            if nuclear > delta:
                obj_c += nuclear
            
                # discard small singular values
                r = s > eigThresh
                uprod = torch.matmul(v[:, r], torch.transpose(u[:, r], 0, 1))
            
                d_z_geometry[class_indices] += uprod
            else:
                obj_c += delta

        v, s, u = torch.svd(z_geometry)

        r = s > eigThresh
        obj = (obj_c - torch.sum(s))/N
        d_z_geometry = (d_z_geometry - torch.matmul(v[:, r], torch.transpose(u[:, r], 0, 1)))/N

        ctx.save_for_backward(d_z_geometry)
        return obj

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        d_z_geometry, = ctx.saved_tensors

        return grad_output*d_z_geometry, None, None, None


class OLEGRSV(torch.nn.Module):
    r"""Compute a geometrically-validated loss using orthogonal subspaces determined by SVD

    Arguments:
        classes (list): list of class labels found in the truth labels
        l (float): scale factor for the classification loss
        class_loss (nn.Module): loss function for classification (defaults to cross entropy)
        delta (float): lower limit of sum of eigenvalues
        eigThresh (float): lower limit on eigenvalues to keep
        min_singular_value_fraction (float): each category must have singular values greater than this fraction of the total
        svd_grad (bool): backpropagate through the SVD operation (O(N^4) complexity)

    Attributes:
        l (float): scale factor for the classification loss
        svd_grad (bool): backpropagate through the SVD operation (O(N^4) complexity)
    """
    def __init__(self,
                 classes,
                 l=1.0,
                 class_loss=None,
                 delta=1.0, eigThresh=1e-6, min_singular_value_fraction=0.1,
                 svd_grad=False):
        super(OLEGRSV, self).__init__()
        self._classes = torch.Tensor(classes)
        self._n_classes = self._classes.numel()
        self.l = l
        self._class_loss = class_loss
        self._delta = delta
        self._eigThresh = eigThresh
        self._epsilon = torch.Tensor([eigThresh])
        self._float_zero = torch.Tensor([0.0])
        self._min_singular_value_fraction = min_singular_value_fraction
        self.svd_grad = svd_grad
        self._ole = _OLE.apply

    def forward(self, z_geometry, y_geometry, z_validation, y_validation):
        r"""Computes the OLE-GRSV loss.

        Arguments:
            z_geometry (Tensor): features which determine the geometry;
                assumed to have shape (n-vectors, d-features)
            y_geometry (Tensor): labels of the geometry features
            z_validation (Tensor): features which are validated against the
                the subspace spanned by `z_geometry`
            y_validation (Tensor): labels of the validation features
        
        Returns:
            loss (Tensor): OLE loss plus the class loss
        """
        y_pred_c, z_geometry_svd = self.predict(z_geometry, y_geometry, z_validation, True)

        if self._class_loss is None:
            class_obj = -torch.mean(torch.log(torch.gather(y_pred_c, 1, torch.unsqueeze(y_validation, dim=1))))
        else:
            class_obj = self._class_loss(y_pred_c, y_validation)

        ole_obj = self._ole(z_geometry, z_geometry_svd, self._delta, self._eigThresh)

        return ole_obj + self.l*class_obj

    def predict(self, z_geometry, y_geometry, z_validation, return_svd=False):
        r"""Predict probabilities of each category using orthogonal subspaces

        Arguments:
            z_geometry (Tensor): features which determine the geometry;
                assumed to have shape (n-vectors, d-features)
            y_geometry (Tensor): labels of the geometry features
            z_validation (Tensor): features which are validated against the
                the subspace spanned by `z_geometry`
            y_validation (Tensor): labels of the validation features
            return_svd (bool): whether or not to return the SVD of the geometry batch
        
        Returns:
            y_pred (Tensor): predicted probabilities of each category
        """
        device = y_geometry.device
        epsilon = self._epsilon.to(device)
        float_zero = self._float_zero.to(device)
        # classes_geometry = torch.unique(y_geometry.cpu(), sorted=True).to(device)
        classes_geometry = self._classes.to(device=device, dtype=y_geometry.dtype)

        projs_geometry_validation = []
        z_geometry_svd = []
        for class_geometry in classes_geometry:
            class_indices = y_geometry == class_geometry
            if torch.sum(class_indices) == 0:
                projs_geometry_validation.append(torch.zeros((self._n_classes)))
                continue
            z_geometry_c = z_geometry[class_indices]
            # NOTE: the odd notation here (v,s,u rather than u,s,v) is due to the
            #       misalignment between the notation in the paper and how the
            #       features are formatted in Pytorch (it avoids needless matrix
            #       transposes)
            try:
                if self.svd_grad:
                    v, s, u = torch.svd(z_geometry_c)
                else:
                    with torch.no_grad():
                        v, s, u = torch.svd(z_geometry_c)
            except RuntimeError:
                print('SVD may have failed - check the input features')
                print('class index: {}'.format(class_geometry))
                print('features:\n{}'.format(z_geometry_c))
                raise

            valid_subspace_indices = s.ge(self._eigThresh)
            if torch.sum(valid_subspace_indices) == 0:
                projs_geometry_validation.append(torch.zeros((self._n_classes)))
                continue
            u = u[:, valid_subspace_indices]
            s = s[valid_subspace_indices]
            v = v[:, valid_subspace_indices]
            z_geometry_svd.append([v, s, u, class_indices])

            s_norm = s/torch.max(s)
            valid_subspace_indices = s_norm.ge(self._min_singular_value_fraction)
            if torch.sum(valid_subspace_indices) == 0:
                projs_geometry_validation.append(torch.zeros((self._n_classes)))
                continue
            u = u[:, valid_subspace_indices]
            s = s[valid_subspace_indices]
            v = v[:, valid_subspace_indices]

            z_validation_proj_c = torch.matmul(z_validation, torch.matmul(u, u.transpose(0, 1)))
            # normalize projections
            z_validation_proj_c = z_validation_proj_c/torch.max(torch.norm(z_validation_proj_c, p=2, dim=1, keepdim=True), other=epsilon)

            # NOTE: the `max` here is required because numerical instabilities can cause small
            #       negative values to corrupt the subspaces and ultimately ruin the (already
            #       unstable) SVD computation (this inner product should NEVER be negative)
            z_validation_inner_product = torch.sum(torch.max(z_validation*z_validation_proj_c, other=float_zero), dim=1)
            projs_geometry_validation.append(z_validation_inner_product)

        projs_geometry_validation = torch.stack(projs_geometry_validation)
        projs_geometry_validation = projs_geometry_validation/torch.sum(projs_geometry_validation, dim=0, keepdim=True)
        y_pred_c = projs_geometry_validation.transpose(0, 1)

        if return_svd:
            return y_pred_c, z_geometry_svd
        else:
            return y_pred_c

    def predict_from_projections(self, projection_matrices, z):
        r"""Predict probabilities of each category using orthogonal subspaces

        Arguments:
            projection_matrices (Tensor): features which determine the geometry;
                assumed to have shape (n-vectors, d-features)
            z (Tensor): labels of the geometry features
        
        Returns:
            y_pred (Tensor): predicted probabilities of each category
        """
        device = z.device
        epsilon = self._epsilon.to(device)
        float_zero = self._float_zero.to(device)

        projs_geometry_validation = []
        for i, _ in enumerate(self._classes):
            projection_matrix = projection_matrices[i]
            z_proj = torch.matmul(z, projection_matrix)
            z_proj = z_proj/torch.max(torch.norm(z_proj, p=2, dim=1, keepdim=True), other=epsilon)

            z_inner_product = torch.sum(torch.max(z*z_proj, other=float_zero), dim=1)
            projs_geometry_validation.append(z_inner_product)

        projs_geometry_validation = torch.stack(projs_geometry_validation)
        projs_geometry_validation = projs_geometry_validation/torch.sum(projs_geometry_validation, dim=0, keepdim=True)
        return projs_geometry_validation.transpose(0, 1)