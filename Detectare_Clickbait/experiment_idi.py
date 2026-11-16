import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

def calculate_idi(title, paragraphs):
    # daca nu avem paragrafe sau e doar unul, nu putem calcula amanarea
    if not paragraphs or len(paragraphs) < 2:
        return 0.0, 0, 0.0
    
    # combinam titlul si paragrafele intr-o singura lista pentru vectorizare
    texts_to_analyze = [title] + paragraphs
    
    # initializam vectorizatorul tf-idf, stergem cuvintele de legatura in engleza
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # transformam textele in numere (matrice matematica)
    tfidf_matrix = vectorizer.fit_transform(texts_to_analyze)
    
    title_vector = tfidf_matrix[0:1]
    paragraph_vectors = tfidf_matrix[1:]
    
    # calculam similaritatea cosinus
    similarities = cosine_similarity(title_vector, paragraph_vectors)[0]
    
    # gasim pozitia paragrafului cu cel mai mare scor de similaritate
    max_index = np.argmax(similarities)
    
    # calculam indexul de amanare (idi)
    total_paragraphs = len(paragraphs) - 1
    idi_score = max_index / total_paragraphs if total_paragraphs > 0 else 0.0
    
    return idi_score, max_index, similarities[max_index]

def run_case_study(data_folder, num_samples=10):
    truth_file = os.path.join(data_folder, 'truth', 'majority.csv')
    problems_folder = os.path.join(data_folder, 'problems')
    
    labels_df = pd.read_csv(truth_file, names=['id', 'label'])
    results = []
    processed_count = 0
    
    print("Incepem studiul de caz cu filtrare imbunatatita a continutului...\n")
    
    for index, row in labels_df.iterrows():
        article_id = str(row['id'])
        actual_label = row['label']
        
        html_path = os.path.join(problems_folder, article_id, f"{article_id}.html")
        if not os.path.exists(html_path):
            continue
            
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else "Fara Titlu"
            
            # incercam sa izolam doar corpul articolului pentru a evita reclamele si footer-ul
            article_body = soup.find('article')
            search_area = article_body if article_body else soup
            
            paragraphs = []
            for p in search_area.find_all('p'):
                text = p.get_text(strip=True)
                # crestem pragul la 50 de caractere pentru a filtra meniurile si notele scurte
                if len(text) > 50:
                    paragraphs.append(text)
            
            if len(paragraphs) < 2:
                continue
                
            idi_score, max_idx, max_sim = calculate_idi(title, paragraphs)
            
            results.append({
                'id': article_id,
                'titlu': title,
                'eticheta_reala': actual_label,
                'scor_idi': round(idi_score, 2),
                'similaritate_max': round(max_sim, 2),
                'paragraf_maxim': max_idx + 1,
                'total_paragrafe': len(paragraphs)
            })
            
            processed_count += 1
            if processed_count >= num_samples:
                break

    print("-" * 65)
    for res in results:
        print(f"Titlu: {res['titlu'][:60]}...")
        print(f"Eticheta Webis: {res['eticheta_reala']}")
        print(f"Scor IDI: {res['scor_idi']} | Sim. Max: {res['similaritate_max']} (paragraf {res['paragraf_maxim']} din {res['total_paragrafe']})")
        print("-" * 65)

if __name__ == "__main__":
    run_case_study(data_folder="data", num_samples=5)