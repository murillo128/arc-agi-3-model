"""Fixed-plane parameter trajectories on explicitly sampled coordinates (CPU)."""
import numpy as np


class ParameterPlane:
    def __init__(self, samples=None, basis=None, origin=None, basis_id=0):
        self.samples = list(samples) if samples is not None else []
        self.basis, self.origin, self.basis_id = basis, origin, basis_id

    def append(self, vector):
        self.samples.append(np.asarray(vector, dtype=np.float32))
        self.samples = self.samples[-96:]
        if self.basis is None and len(self.samples) >= 4:
            self.calibrate()

    def calibrate(self):
        if len(self.samples) < 3:
            return False
        x = np.stack(self.samples[-32:]).astype(np.float64)
        self.origin = x.mean(0)
        centered = x - self.origin
        # Eigenvectors of the small temporal Gram matrix, not an NxN parameter covariance.
        eigenvalues, vectors = np.linalg.eigh(centered @ centered.T)
        axes = []
        for j in np.argsort(eigenvalues)[::-1][:2]:
            if eigenvalues[j] <= max(1e-24, eigenvalues[-1] * 1e-10):
                axes.append(np.zeros(x.shape[1]))
            else:
                axis = vectors[:, j] @ centered / np.sqrt(eigenvalues[j])
                if axis[np.argmax(abs(axis))] < 0:
                    axis = -axis
                axes.append(axis)
        self.basis = np.stack(axes)
        self.basis_id += 1
        return True

    def project(self):
        if self.basis is None:
            return []
        x = np.stack(self.samples).astype(np.float64)
        xy = (x - self.origin) @ self.basis.T
        result, previous = [], None
        for i, point in enumerate(xy):
            delta = x[i] - x[i-1] if i else None
            norm = np.linalg.norm(delta) if delta is not None else 0.
            coverage = float(np.clip(np.sum((self.basis @ delta)**2) / norm**2, 0, 1)) if norm else None
            angle = None
            if norm and previous is not None and np.linalg.norm(previous):
                cosine = np.dot(delta, previous) / (norm * np.linalg.norm(previous))
                angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            result.append(dict(x=float(point[0]), y=float(point[1]), coverage=coverage,
                               step_norm=float(norm), angle=angle))
            previous = delta
        return result
