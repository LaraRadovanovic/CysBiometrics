import numpy as np
import math
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from performance import Evaluator

def extract_face_features(X):
    N = X.shape[0]
    feats = []
    for pid in range(N):
        lm = X[pid]
        center = lm.mean(axis=0)
        w, h = np.ptp(lm[:, 0]), np.ptp(lm[:, 1])
        v = []
        for i in range(len(lm)):
            for j in range(i+1, len(lm)):
                p1, p2 = lm[i], lm[j]
                d = np.linalg.norm(p1 - p2)
                a = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                r = d / (abs(p1[0] - p2[0]) + 1e-6)
                s = abs(p1[0] - p2[0])
                disp = np.linalg.norm(p1 - center)
                rel = d / (w + h + 1e-6)
                ad = abs(math.atan2(center[1] - p1[1], center[0] - p1[0]) - math.atan2(center[1] - p2[1], center[0] - p2[0]))
                sr = s / (w + 1e-6)
                v.extend([d, a, r, s, disp, rel, ad, sr])
        feats.append(v)
    return np.array(feats)

def extract_voice_features(X_wavs, sr=16000, n_mfcc=13):
    feats = []
    for wav in X_wavs:
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
        feats.append(np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.vstack(feats)

def evaluate_scores(scores, y_te, title):
    genuine, impostor = [], []
    for i, label in enumerate(y_te):
        genuine.append(scores[i, label])
        for c in range(scores.shape[1]):
            if c != label:
                impostor.append(scores[i, c])
    evaluator = Evaluator(200, genuine, impostor, title)
    FPR, FNR, TPR = evaluator.compute_rates()
    evaluator.plot_score_distribution()
    evaluator.plot_det_curve(FPR, FNR)
    evaluator.plot_roc_curve(FPR, TPR)
    eer = evaluator.get_EER(FPR, FNR)
    dprime = evaluator.get_dprime()
    return FPR, FNR, TPR, eer, dprime

def plot_roc_and_det(results):
    colors = {
        "Face Only": "blue",
        "Voice Only": "red",
        "Face + Voice": "purple"
    }

    # --- ROC Curve ---
    fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
    for i, (name, data) in enumerate(results.items()):
        ax_roc.plot(data["FPR"], data["TPR"], label=f"{name} (AUC = {data['AUC']:.4f})", color=colors[name])
        
        # Key values
        eer = data['EER']
        far_idx = np.argmin(np.abs(np.array(data['FPR']) - 0.01))
        far_val = data['FPR'][far_idx]
        tar_val = data['TPR'][far_idx]

        # Annotation
        annotation = (
            f"EER: {eer:.4f}\n"
            f"FAR@0.01: {far_val:.4f}\n"
            f"TAR@0.01: {tar_val:.4f}"
        )
        ax_roc.text(1.05, 0.8 - i*0.2, annotation, transform=ax_roc.transAxes,
                    fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'),
                    verticalalignment='top', color=colors[name])

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.5)
    ax_roc.set_xlabel("False Accept Rate (FAR)", fontsize=12, weight='bold')
    ax_roc.set_ylabel("True Accept Rate (TAR)", fontsize=12, weight='bold')
    ax_roc.set_title("ROC Curve - Face, Voice, and Fused", fontsize=14, weight='bold')
    ax_roc.grid(True)
    ax_roc.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig("ROC_Curve_All.png", dpi=300)
    plt.show()

    # --- DET Curve ---
    fig_det, ax_det = plt.subplots(figsize=(10, 6))
    for i, (name, data) in enumerate(results.items()):
        ax_det.plot(data["FPR"], data["FNR"], label=f"{name} (EER = {data['EER']:.4f})", color=colors[name])
        
        # Key values
        eer = data['EER']
        far_idx = np.argmin(np.abs(np.array(data['FPR']) - 0.01))
        far_val = data['FPR'][far_idx]
        frr_val = data['FNR'][far_idx]

        # Annotation
        annotation = (
            f"EER: {eer:.4f}\n"
            f"FAR@0.01: {far_val:.4f}\n"
            f"FRR@0.01: {frr_val:.4f}"
        )
        ax_det.text(1.05, 0.8 - i*0.2, annotation, transform=ax_det.transAxes,
                    fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'),
                    verticalalignment='top', color=colors[name])

    ax_det.plot([0, 1], [0, 1], 'k--', lw=0.5)
    ax_det.set_xlabel("False Accept Rate (FAR)", fontsize=12, weight='bold')
    ax_det.set_ylabel("False Reject Rate (FRR)", fontsize=12, weight='bold')
    ax_det.set_title("DET Curve - Face, Voice, and Fused", fontsize=14, weight='bold')
    ax_det.grid(True)
    ax_det.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig("DET_Curve_All.png", dpi=300)
    plt.show()

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
    idx = np.random.choice(len(X_voice), size=len(X_face), replace=False)
    X_voice_sel = X_voice[idx]

    F = extract_face_features(X_face)
    V = extract_voice_features(X_voice_sel)

    ids = np.arange(len(y_face_enc))
    train_ids, test_ids = train_test_split(ids, test_size=0.33, stratify=y_face_enc, random_state=42)
    F_tr, F_te = F[train_ids], F[test_ids]
    V_tr, V_te = V[train_ids], V[test_ids]
    y_tr, y_te = y_face_enc[train_ids], y_face_enc[test_ids]

    scaler_F = StandardScaler().fit(F_tr)
    scaler_V = StandardScaler().fit(V_tr)
    F_tr, F_te = scaler_F.transform(F_tr), scaler_F.transform(F_te)
    V_tr, V_te = scaler_V.transform(V_tr), scaler_V.transform(V_te)

    pca_F = PCA(n_components=0.95).fit(F_tr)
    pca_V = PCA(n_components=0.95).fit(V_tr)
    F_tr, F_te = pca_F.transform(F_tr), pca_F.transform(F_te)
    V_tr, V_te = pca_V.transform(V_tr), pca_V.transform(V_te)

    clf_face = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)).fit(F_tr, y_tr)
    clf_voice = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)).fit(V_tr, y_tr)

    scores_face = clf_face.predict_proba(F_te)
    scores_voice = clf_voice.predict_proba(V_te)
    scores_fused = 0.6 * scores_face + 0.4 * scores_voice

    FPR_f, FNR_f, TPR_f, eer_f, dp_f = evaluate_scores(scores_face, y_te, "Face Only")
    FPR_v, FNR_v, TPR_v, eer_v, dp_v = evaluate_scores(scores_voice, y_te, "Voice Only")
    FPR_s, FNR_s, TPR_s, eer_s, dp_s = evaluate_scores(scores_fused, y_te, "Face + Voice")

    results = {
    "Face Only": {
        "FPR": FPR_f, "TPR": TPR_f, "FNR": FNR_f,
        "EER": eer_f,
        "AUC": metrics.auc(FPR_f, TPR_f)
    },
    "Voice Only": {
        "FPR": FPR_v, "TPR": TPR_v, "FNR": FNR_v,
        "EER": eer_v,
        "AUC": metrics.auc(FPR_v, TPR_v)
    },
    "Face + Voice": {
        "FPR": FPR_s, "TPR": TPR_s, "FNR": FNR_s,
        "EER": eer_s,
        "AUC": metrics.auc(FPR_s, TPR_s)
    }
}

    plot_roc_and_det(results)

if __name__ == "__main__":
    main()
