import numpy as np
from nltk.tree import Tree
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coords_and_feats):
        for t in self.transforms:
            coords_and_feats = t(coords_and_feats)
        return coords_and_feats

    def __str__(self) -> str:
        sep = "\n\t"
        msg = "Compose of ["
        for t in self.transforms:
            msg += f"{sep}{t},"
        msg += "]\n"
        return msg

    def __repr__(self) -> str:
        sep = "\n\t"
        msg = "Compose of ["
        for t in self.transforms:
            msg += f"{sep}{t}"
        msg += "]\n"
        return msg


class RandomDropSubTrees(object):
    def __init__(self, probs=[0.1, 0.2, 0.3, 0.4, 0.5], max_cnt=5):
        self.probs = probs
        self.max_cnt = max_cnt
        self.cnt = 0

    def remove_subtrees(self, root, level_idx):
        reduced_root = Tree(root.label(), [])
        if len(root) == 0:
            return reduced_root
        p = np.random.uniform(0, 1, len(root))
        # padding the dropping probabilities
        if level_idx >= len(self.probs):
            level_idx = len(self.probs) - 1
        for child_idx in range(len(root)):
            if self.cnt > self.max_cnt:
                reduced_root.append(root[child_idx])
                continue
            else:
                if p[child_idx] > self.probs[level_idx]:
                    reduced_root.append(
                        self.remove_subtrees(root[child_idx], level_idx + 1)
                    )
                else:
                    self.cnt += 1
        return reduced_root

    def __call__(self, tree):
        self.cnt = 0
        return self.remove_subtrees(tree, level_idx=0)

    def __str__(self) -> str:
        return f"RandomDropSubTrees(probs={self.probs}, max_cnt={self.max_cnt})"

    def __repr__(self) -> str:
        return f"RandomDropSubTrees(probs={self.probs}, max_cnt={self.max_cnt})"


class RandomSkipParentNode(object):
    def __init__(self, probs=[0.1], max_cnt=5):
        self.probs = probs
        self.max_cnt = max_cnt
        self.cnt = 0

    def move_grandson_to_son(self, root, level_idx):
        if len(root) == 0 or len(root) == 1:
            return root
        p = np.random.uniform(0, 1, len(root))
        # padding the dropping probabilities
        if level_idx >= len(self.probs):
            level_idx = len(self.probs) - 1
        for child_idx in range(len(root)):
            if self.cnt >= self.max_cnt:
                break
            if p[child_idx] < self.probs[level_idx]:
                if len(root[child_idx]) == 0 or len(root[child_idx]) == 1:
                    continue
                else:
                    idx = random.randint(0, len(root[child_idx]) - 1)
                    root[child_idx] = root[child_idx][idx]
                    self.cnt += 1
            else:
                root[child_idx] = self.move_grandson_to_son(
                    root[child_idx], level_idx + 1
                )
        return root

    def __call__(self, tree):
        self.cnt = 0
        return self.move_grandson_to_son(tree, level_idx=0)

    def __str__(self) -> str:
        return f"RandomSkipParentNode(probs={self.probs}, max_cnt={self.max_cnt})"

    def __repr__(self) -> str:
        return f"RandomSkipParentNode(probs={self.probs}, max_cnt={self.max_cnt})"


class RandomSwapSiblingSubTrees(object):
    def __init__(self, probs=[0.1], max_cnt=5):
        self.probs = probs
        self.max_cnt = max_cnt
        self.cnt = 0

    def swap_sibling_subtrees(self, root, level_idx):
        if len(root) < 2:
            return root
        p = np.random.uniform(0, 1, len(root))
        # padding the dropping probabilities
        if level_idx >= len(self.probs):
            level_idx = len(self.probs) - 1
        for child_idx in range(len(root)):
            if self.cnt >= self.max_cnt:
                break
            if p[child_idx] < self.probs[level_idx]:
                if len(root[child_idx]) < 2:
                    continue
                else:
                    my_subtree_idx = random.randint(0, len(root[child_idx]) - 1)
                    sibling_idx = random.randint(0, len(root) - 1)
                    if len(root[sibling_idx]) == 0:
                        continue
                    sibling_subtree_idx = random.randint(0, len(root[sibling_idx]) - 1)
                    my_subtree = root[child_idx][my_subtree_idx].copy()
                    sibling_subtree = root[sibling_idx][sibling_subtree_idx].copy()
                    root[child_idx][my_subtree_idx] = sibling_subtree
                    root[sibling_idx][sibling_subtree_idx] = my_subtree
                    self.cnt += 1
            else:
                root[child_idx] = self.swap_sibling_subtrees(
                    root[child_idx], level_idx + 1
                )
        return root

    def __call__(self, tree):
        self.cnt = 0
        return self.swap_sibling_subtrees(tree, level_idx=0)

    def __str__(self) -> str:
        return f"RandomSwapSiblingSubTrees(probs={self.probs}, max_cnt={self.max_cnt})"

    def __repr__(self) -> str:
        return f"RandomSwapSiblingSubTrees(probs={self.probs}, max_cnt={self.max_cnt})"


class RandomRotateAligned(object):
    def __init__(self, p=0.5, axis=2):
        self.prob = p
        self.axis = axis

    def __call__(self, coords_and_feats):
        coord = coords_and_feats[:, :3]
        if np.random.rand() < self.prob:
            angle = np.random.uniform() * 2 * np.pi
            cos, sin = np.cos(angle), np.sin(angle)
            R_x = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
            R_y = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
            R_z = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
            R = [R_x, R_y, R_z][self.axis]
            coord = np.dot(coord, R)
        coords_and_feats[:, :3] = coord
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomRotateAligned(p={self.prob},axis={self.axis})"

    def __repr__(self) -> str:
        return f"RandomRotateAligned(p={self.prob},axis={self.axis})"


class RandomRotate(object):
    def __init__(self, sigma=0.03, clip=0.09, p=0.5):
        self.sigma = sigma
        self.clip = clip
        self.prob = p

    def __call__(self, coords_and_feats):
        coord = coords_and_feats[:, :3]
        if np.random.rand() < self.prob:
            angle_x = np.random.uniform() * 2 * np.pi
            angle_y = np.random.uniform() * 2 * np.pi
            angle_z = np.random.uniform() * 2 * np.pi
            cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
            cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
            R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            R = np.dot(R_z, np.dot(R_y, R_x))
            coord = np.dot(coord, R)
        coords_and_feats[:, :3] = coord
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomRotate(p={self.prob})"

    def __repr__(self) -> str:
        return f"RandomRotate(p={self.prob})"


class RandomMaskFeats(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, coords_and_feats):
        if len(coords_and_feats[0]) > 5:
            feats = coords_and_feats[:, 5:]
            feats[
                :,
                np.random.choice(
                    np.arange(len(feats[0])), int(len(feats[0]) * self.prob)
                ),
            ] = 0
            coords_and_feats[:, 5:] = feats
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomMaskFeats(p={self.prob})"

    def __repr__(self) -> str:
        return f"RandomMaskFeats(p={self.prob})"


class RandomElasticate(object):
    def __init__(self, p=0.2, scales=[0.8, 1.2]):
        self.prob = p
        self.scales = scales

    def __call__(self, coords_and_feats):
        if len(coords_and_feats[0]) > 5:
            if np.random.rand() < self.prob:
                branches = coords_and_feats[:, 5:]
                scales = np.random.uniform(
                    self.scales[0], self.scales[1], branches.shape
                )
                branches *= scales
                coords_and_feats[:, 5:] = branches
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomElasticate(p={self.prob}, scales={self.scales})"

    def __repr__(self) -> str:
        return f"RandomElasticate(p={self.prob}, scales={self.scales})"


class RandomScaleCoords(object):
    def __init__(self, scale=[0.8, 1.2], p=0.5):
        self.scale = scale
        self.prob = p

    def __call__(self, coords_and_feats):
        if np.random.rand() < self.prob:
            scale = np.random.uniform(self.scale[0], self.scale[1])
            coords_and_feats[:, :4] *= scale
            if len(coords_and_feats[0]) > 5:
                coords_and_feats[:, 5:] *= scale
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomScaleCoords(p={self.prob}, scale={self.scale})"

    def __repr__(self) -> str:
        return f"RandomScaleCoords(p={self.prob}, scale={self.scale})"


class RandomScaleCoordsTranslation(object):
    def __init__(self, scale=[0.5, 2], p=0.5):
        self.scale = scale
        self.prob = p

    def __call__(self, coords_and_feats):
        if np.random.rand() < self.prob:
            scale = np.random.uniform(self.scale[0], self.scale[1])
            coord1 = coords_and_feats[:, :4]
            coord1 *= scale
            coords_and_feats[:, :4] = coord1
            if len(coords_and_feats[0]) > 5:
                coord2 = coords_and_feats[:, 5:]
                coord2 *= scale
                coords_and_feats[:, 5:] = coord2
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomScaleCoordsTranslation(p={self.prob}, scale={self.scale})"

    def __repr__(self) -> str:
        return f"RandomScaleCoordsTranslation(p={self.prob}, scale={self.scale})"


class RandomScaleFeats(object):
    def __init__(self, scale=[0.5, 2], p=0.5):
        self.scale = scale
        self.prob = p

    def __call__(self, coords_and_feats):
        feats = coords_and_feats[:, 4:]
        if np.random.rand() < self.prob:
            scale = np.random.uniform(self.scale[0], self.scale[1])
            feats *= scale
        coords_and_feats[:, 4:] = feats
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomScaleFeats(p={self.prob}, scale={self.scale})"

    def __repr__(self) -> str:
        return f"RandomScaleFeats(p={self.prob}, scale={self.scale})"


class RandomShift(object):
    def __init__(self, shift=[5, 5, 5], p=0.5):
        self.shift = shift
        self.prob = p

    def __call__(self, coords_and_feats):
        coord = coords_and_feats[:, :3]
        if np.random.rand() < self.prob:
            shift_x = np.random.uniform(-self.shift[0], self.shift[0])
            shift_y = np.random.uniform(-self.shift[1], self.shift[1])
            shift_z = np.random.uniform(-self.shift[2], self.shift[2])
            coord += [shift_x, shift_y, shift_z]
        coords_and_feats[:, :3] = coord
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomShift(p={self.prob}, shift={self.shift})"

    def __repr__(self) -> str:
        return f"RandomShift(p={self.prob}, shift={self.shift})"


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, coords_and_feats):
        coord = coords_and_feats[:, :3]
        if np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                coord[:, 0] = -coord[:, 0]
            if np.random.rand() < 0.5:
                coord[:, 1] = -coord[:, 1]
        coords_and_feats[:, :3] = coord
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomFlip(p={self.prob})"

    def __repr__(self) -> str:
        return f"RandomFlip(p={self.prob})"


class RandomJitter(object):
    def __init__(self, sigma=1, clip=5, p=0.5):
        self.sigma = sigma
        self.clip = clip
        self.prob = p

    def __call__(self, coords_and_feats):
        coord = coords_and_feats[:, :3]
        assert self.clip > 0
        if np.random.rand() < self.prob:
            jitter = np.clip(
                self.sigma * np.random.randn(coord.shape[0], 3), -self.clip, self.clip
            )
            coord += jitter
        coords_and_feats[:, :3] = coord
        return coords_and_feats

    def __str__(self) -> str:
        return f"RandomJitter(p={self.prob}, sigma={self.sigma}, clip={self.clip})"

    def __repr__(self) -> str:
        return f"RandomJitter(p={self.prob}, sigma={self.sigma}, clip={self.clip})"


class RandomJitterLength(object):
    def __init__(self, sigma=0.1, clip=1, p=0.5):
        self.sigma = sigma
        self.clip = clip
        self.prob = p

    def __call__(self, coords_and_feats):
        feats1 = coords_and_feats[:, 3:4]
        assert self.clip > 0
        if np.random.rand() < self.prob:
            jitter1 = np.clip(
                self.sigma * np.random.randn(*feats1.shape), -self.clip, self.clip
            )
            feats1 += jitter1
            if len(coords_and_feats[0]) > 5:
                feats2 = coords_and_feats[:, 5:]
                jitter2 = np.clip(
                    self.sigma * np.random.randn(*feats2.shape), -self.clip, self.clip
                )
                feats2 += jitter2
                coords_and_feats[:, 5:] = feats2
        coords_and_feats[:, 3:4] = feats1
        return coords_and_feats

    def __str__(self) -> str:
        return (
            f"RandomJitterLength(p={self.prob}, sigma={self.sigma}, clip={self.clip})"
        )

    def __repr__(self) -> str:
        return (
            f"RandomJitterLength(p={self.prob}, sigma={self.sigma}, clip={self.clip})"
        )


if __name__ == "__main__":
    transform = RandomScaleCoordsTranslation(p=1)

    coords_feats = np.random.rand(1024, 3)
    # coords_feats = np.concatenate([
    #     np.zeros((1,29)),
    #     coords_feats
    # ])
    print(coords_feats[-1])
    transformed = transform(coords_feats)
    print(transformed[-1])
    print(coords_feats[-1])
