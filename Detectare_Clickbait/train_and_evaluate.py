import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_models(csv_path="webis_features.csv"):
    # incarcam setul de trasaturi deja extras
    df = pd.read_csv(csv_path)
    
    # selectam coloanele de intrare (trasaturile calculate) si eticheta tinta
    feature_cols = [
        'idi_score',
        'max_similarity',
        'max_paragraph_idx',
        'total_paragraphs',
        'title_word_count',
        'has_question_mark'
    ]
    X = df[feature_cols]
    y = df['label']
    
    # impartim setul in 80% antrenare si 20% testare, pastrand proportia claselor
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Total articole analizate: {len(df)}")
    print(f"Set antrenare (80%): {len(X_train)} | Set testare (20%): {len(X_test)}\n")
    
    # standardizam valorile numerice (necesar pentru Logistic Regression si SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # definim cei 3 clasificatori 
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', class_weight='balanced', random_state=42)
    }
    
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision (CB)':<14} | {'Recall (CB)':<11} | {'F1 (CB)':<8} | {'Macro F1':<8}")
    print("-" * 85)
    
    best_model = None
    best_f1 = -1.0
    best_name = ""
    
    for name, model in models.items():
        # antrenam modelul pe datele scalate
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec_cb = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_cb = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_cb = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"{name:<22} | {acc:<8.4f} | {prec_cb:<14.4f} | {rec_cb:<11.4f} | {f1_cb:<8.4f} | {macro_f1:<8.4f}")
        
        if f1_cb > best_f1:
            best_f1 = f1_cb
            best_model = model
            best_name = name

    # afisam importanta trasaturilor pentru Random Forest (ca sa vedem cat cantareste IDI)
    rf_model = models["Random Forest"]
    print("\nImportanta trasaturilor (Random Forest):")
    for col, imp in sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  - {col:<20}: {imp:.4f}")

    # raport detaliat pentru cel mai bun model
    print(f"\nRaport detaliat pentru cel mai bun model ({best_name}):")
    y_best_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_best_pred, target_names=['no-clickbait', 'clickbait'], digits=4))

if __name__ == "__main__":
    evaluate_models()