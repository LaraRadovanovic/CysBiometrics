import numpy as np
import math
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from performance import Evaluator

def extract_face_features(X, feature_types=['euclidean', 'relative', 'ratio']):
    N = X.shape[0]
    feats = []
    for pid in range(N):
        lm = X[pid]
        w, h = np.ptp(lm[:, 0]), np.ptp(lm[:, 1])
        face_size = w + h + 1e-6
        v = []
        for i in range(len(lm)):
            for j in range(i + 1, len(lm)):
                p1, p2 = lm[i], lm[j]
                d = np.linalg.norm(p1 - p2)
                h_dist = abs(p1[0] - p2[0]) + 1e-6
                if 'euclidean' in feature_types:
                    v.append(d)
                if 'relative' in feature_types:
                    v.append(d / face_size)
                if 'ratio' in feature_types:
                    v.append(d / h_dist)
        feats.append(v)
    return np.array(feats)

def extract_voice_features(X_wavs, sr=16000, n_mfcc=13):
    feats = []
    for wav in X_wavs:
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
        feats.append(np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.vstack(feats)

def main():
    # Load data
    X_face = np.load("X-68-Caltech.npy")
    y_face = np.load("y-68-Caltech.npy")
    X_voice = np.load("X-voice.npy")
    y_voice_int = np.load("y-voice.npy")
    voice_classes = np.load("y-voice_classes.npy")
    y_voice_str = voice_classes[y_voice_int]

    le = LabelEncoder().fit(np.concatenate([y_face, y_voice_str]))
    y_face_enc = le.transform(y_face)

    np.random.seed(42)
    voice_indices = np.random.choice(len(X_voice), size=len(X_face), replace=False)
    X_voice_sel = X_voice[voice_indices]

    # Extract voice features once
    V = extract_voice_features(X_voice_sel)

    # Train/test split
    ids = np.arange(len(y_face_enc))
    train_ids, test_ids = train_test_split(ids, test_size=0.33, stratify=y_face_enc, random_state=42)
    V_tr, V_te = V[train_ids], V[test_ids]
    y_tr, y_te = y_face_enc[train_ids], y_face_enc[test_ids]

    # Face feature sets to test
    face_feature_sets = [
        ['euclidean'],
        ['relative'],
        ['ratio'],
        ['euclidean', 'relative'],
        ['euclidean', 'ratio'],
        ['relative', 'ratio'],
        ['euclidean', 'relative', 'ratio']
    ]

    for feature_types in face_feature_sets:
        print(f"\n Testing Fusion with Face Features: {feature_types}")

        F = extract_face_features(X_face, feature_types=feature_types)
        F_tr, F_te = F[train_ids], F[test_ids]

        # Scale & PCA
        scaler_F = StandardScaler().fit(F_tr)
        scaler_V = StandardScaler().fit(V_tr)
        F_tr_s, F_te_s = scaler_F.transform(F_tr), scaler_F.transform(F_te)
        V_tr_s, V_te_s = scaler_V.transform(V_tr), scaler_V.transform(V_te)

        pca_F = PCA(n_components=0.95).fit(F_tr_s)
        pca_V = PCA(n_components=0.95).fit(V_tr_s)
        F_tr_p, F_te_p = pca_F.transform(F_tr_s), pca_F.transform(F_te_s)
        V_tr_p, V_te_p = pca_V.transform(V_tr_s), pca_V.transform(V_te_s)

        # Classifiers
        face_clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)).fit(F_tr_p, y_tr)
        voice_clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)).fit(V_tr_p, y_tr)
        face_scores = face_clf.predict_proba(F_te_p)
        voice_scores = voice_clf.predict_proba(V_te_p)

        # Fusion
        alpha = 0.6
        fused_scores = alpha * face_scores + (1 - alpha) * voice_scores

        # Evaluate
        genuine, impostor = [], []
        C = fused_scores.shape[1]
        for i, label in enumerate(y_te):
            genuine.append(fused_scores[i, label])
            for c in range(C):
                if c != label:
                    impostor.append(fused_scores[i, c])

        evaluator = Evaluator(
            num_thresholds=200,
            genuine_scores=genuine,
            impostor_scores=impostor,
            plot_title=f"Fusion — Face Features: {feature_types}",
            epsilon=1e-12
        )
        FPR, FNR, TPR = evaluator.compute_rates()
        evaluator.plot_score_distribution()
        evaluator.plot_det_curve(FPR, FNR)
        evaluator.plot_roc_curve(FPR, TPR)

        EER = evaluator.get_EER(FPR, FNR)
        d_prime = evaluator.get_dprime()
        print("EER:", EER)
        print("d-prime:", d_prime)

if __name__ == "__main__":
    main()
