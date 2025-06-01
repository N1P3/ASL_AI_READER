import numpy as np

def extract_features(landmarks_xyz):
    landmarks = np.array(landmarks_xyz).reshape((21, 3))
    center = landmarks[0]
    norm_landmarks = landmarks - center

    # Długości palców
    finger_base_tip_pairs = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    finger_lengths = [np.linalg.norm(landmarks[tip] - landmarks[base]) for base, tip in finger_base_tip_pairs]

    # Odległości między czubkami palców
    fingertip_idxs = [4, 8, 12, 16, 20]
    dists = []
    for i in range(len(fingertip_idxs)):
        for j in range(i + 1, len(fingertip_idxs)):
            d = np.linalg.norm(landmarks[fingertip_idxs[i]] - landmarks[fingertip_idxs[j]])
            dists.append(d)

    # Normalizacja względem rozpiętości dłoni
    spread = np.linalg.norm(landmarks[4] - landmarks[20])
    spread = spread if spread > 0 else 1.0
    finger_lengths_norm = [l / spread for l in finger_lengths]
    dists_norm = [d / spread for d in dists]

    # Kąty między segmentami palców
    def angle_between(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    angle_idxs = [
        (2, 3, 4), (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]
    angles = [angle_between(landmarks[i1], landmarks[i2], landmarks[i3]) for i1, i2, i3 in angle_idxs]

    # Wektor cech
    features = np.concatenate([
        norm_landmarks.flatten(),
        finger_lengths_norm,
        dists_norm,
        angles
    ])
    return features
