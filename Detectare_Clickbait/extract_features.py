import os
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
from experiment_idi import calculate_idi

def extract_all_features(data_folder="data", output_csv="webis_features.csv"):
    # construim caile catre fisierul cu etichete si folderul cu articole
    truth_file = os.path.join(data_folder, 'truth', 'majority.csv')
    problems_folder = os.path.join(data_folder, 'problems')
    
    # incarcam toate etichetele din dataset
    labels_df = pd.read_csv(truth_file, names=['id', 'label'])
    total_rows = len(labels_df)
    print(f"Am gasit {total_rows} inregistrari in majority.csv. Incepem procesarea...\n", flush=True)
    
    # filtram doar etichetele relevante din html pentru o parsare mult mai rapida
    only_relevant_tags = SoupStrainer(['title', 'article', 'p'])
    
    dataset_rows = []
    skipped_missing = 0
    skipped_short = 0
    
    for index, row in labels_df.iterrows():
        article_id = str(row['id'])
        actual_label = row['label']
        
        html_path = os.path.join(problems_folder, article_id, f"{article_id}.html")
        if not os.path.exists(html_path):
            skipped_missing += 1
            continue
            
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as file:
            # parsam strict etichetele filtrate
            soup = BeautifulSoup(file.read(), 'html.parser', parse_only=only_relevant_tags)
            
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else ""
            if not title:
                skipped_short += 1
                continue
                
            # izolam corpul principal al articolului daca exista eticheta article
            article_body = soup.find('article')
            search_area = article_body if article_body else soup
            
            paragraphs = []
            for p in search_area.find_all('p'):
                text = p.get_text(strip=True)
                # pastram doar paragrafele consistente (peste 50 caractere)
                if len(text) > 50:
                    paragraphs.append(text)
                    
            # avem nevoie de cel putin 2 paragrafe pentru a calcula amanarea
            if len(paragraphs) < 2:
                skipped_short += 1
                continue
                
            # calculam scorul idi si similaritatea maxima
            idi_score, max_idx, max_sim = calculate_idi(title, paragraphs)
            
            # extragem trasaturi auxiliare simple din titlu
            title_word_count = len(title.split())
            has_question_mark = 1 if '?' in title else 0
            
            # convertim eticheta text in format binar (1 = clickbait, 0 = no-clickbait)
            numeric_label = 1 if actual_label.strip().lower() == 'clickbait' else 0
            
            dataset_rows.append({
                'id': article_id,
                'idi_score': round(idi_score, 4),
                'max_similarity': round(max_sim, 4),
                'max_paragraph_idx': max_idx + 1,
                'total_paragraphs': len(paragraphs),
                'title_word_count': title_word_count,
                'has_question_mark': has_question_mark,
                'label_text': actual_label,
                'label': numeric_label
            })
            
        # afisam progresul la fiecare 50 de fisiere procesate
        if (index + 1) % 50 == 0:
            print(f"Progres: {index + 1} / {total_rows} fisiere verificate (valide: {len(dataset_rows)})...", flush=True)

    # salvam toate trasaturile extrase in fisierul csv
    features_df = pd.DataFrame(dataset_rows)
    features_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 55, flush=True)
    print(f"Procesare finalizata! Date salvate in: {output_csv}")
    print(f"Articole valide extrase: {len(features_df)}")
    print(f"Fisiere HTML lipsa: {skipped_missing}")
    print(f"Articole eliminate (prea scurte / fara titlu): {skipped_short}")
    print("\nDistributia claselor in setul extras:")
    print(features_df['label_text'].value_counts())
    print("=" * 55, flush=True)

if __name__ == "__main__":
    extract_all_features()