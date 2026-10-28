"""Fixed three-axis parameter trajectories on sampled coordinates (CPU)."""
import numpy as np


class ParameterPlane:
    def __init__(self, samples=None, basis=None, origin=None, basis_id=0):
        self.samples = list(samples) if samples is not None else []
        self.basis, self.origin, self.basis_id = basis, origin, basis_id

    def append(self, vector):
        self.samples.append(np.asarray(vector, dtype=np.float32))
        self.samples = self.samples[-96:]
        # A module may be idle for its first batches (e.g. a sparse planner).
        # A zero basis has no directions to preserve when its first update arrives.
        moving_from_zero = (self.basis is not None and not np.any(self.basis)
                            and len(self.samples) > 1
                            and not np.array_equal(self.samples[-1], self.samples[-2]))
        if len(self.samples) >= 4 and (self.basis is None or moving_from_zero):
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
        for j in np.argsort(eigenvalues)[::-1][:3]:
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
        coordinates = (x - self.origin) @ self.basis.T
        result, previous = [], None
        for i, point in enumerate(coordinates):
            delta = x[i] - x[i-1] if i else None
            norm = np.linalg.norm(delta) if delta is not None else 0.
            squared = (self.basis @ delta)**2 if norm else None
            coverage = float(np.clip(np.sum(squared[:2]) / norm**2, 0, 1)) if norm else None
            angle = None
            if norm and previous is not None and np.linalg.norm(previous):
                cosine = np.dot(delta, previous) / (norm * np.linalg.norm(previous))
                angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            result.append(dict(x=float(point[0]), y=float(point[1]), coverage=coverage,
                               step_norm=float(norm), angle=angle))
            if len(point) == 3:
                result[-1].update(z=float(point[2]), coverage_3d=(
                    float(np.clip(np.sum(squared) / norm**2, 0, 1)) if norm else None))
            previous = delta
        return result
