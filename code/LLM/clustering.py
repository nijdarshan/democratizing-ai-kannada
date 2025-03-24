import pandas as pd
import numpy as np
import torch
from umap import UMAP
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import plotly.io as pio
import os
import gc
import glob

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_color_palette(n_colors):
    """Generate a color palette for visualization"""
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    # If we need more colors than available, cycle through them
    return [colors[i % len(colors)] for i in range(n_colors)]

def remap_cluster_labels(true_labels, predicted_labels):
    """
    Remap cluster labels to best match true labels using Hungarian algorithm
    """
    # Remove samples with label 0 or NaN
    valid_mask = (true_labels != 0) & ~np.isnan(true_labels)
    true_labels_valid = true_labels[valid_mask]
    predicted_labels_valid = predicted_labels[valid_mask]
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels_valid, predicted_labels_valid)
    
    # Use Hungarian algorithm to find optimal mapping
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create mapping dictionary
    label_mapping = {old: new for old, new in zip(col_ind, row_ind)}
    
    # Remap predicted labels
    remapped_labels = np.array([label_mapping.get(label, label) for label in predicted_labels])
    
    return remapped_labels

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate accuracy and precision metrics after removing ignored labels (0) and NaN
    Returns both overall metrics and per-class precision
    """
    # Remove samples with label 0 or NaN
    valid_mask = (true_labels != 0) & ~np.isnan(true_labels)
    true_labels_valid = true_labels[valid_mask]
    predicted_labels_valid = predicted_labels[valid_mask]
    
    # Calculate overall metrics
    accuracy = accuracy_score(true_labels_valid, predicted_labels_valid)
    overall_precision = precision_score(true_labels_valid, predicted_labels_valid, average='weighted')
    
    # Calculate per-class precision
    unique_labels = np.unique(true_labels_valid)
    per_class_precision = {}
    
    for label in unique_labels:
        label_mask = true_labels_valid == label
        true_class = true_labels_valid[label_mask]
        pred_class = predicted_labels_valid[label_mask]
        
        if len(true_class) > 0:
            class_precision = precision_score(
                true_labels_valid == label,
                predicted_labels_valid == label,
                zero_division=0
            )
            per_class_precision[int(label)] = class_precision
    
    return {
        'accuracy': accuracy,
        'overall_precision': overall_precision,
        'per_class_precision': per_class_precision,
        'confusion_matrix': confusion_matrix(true_labels_valid, predicted_labels_valid)
    }

def get_word_embeddings(sentences, words, tokenizer, model):
    print(f"Input sentences: {len(sentences)}")
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    word_embeddings = []
    valid_sentences = []
    valid_words = []

    for idx, (sentence, word) in enumerate(zip(sentences, words)):
        word_tokens = tokenizer.tokenize(word)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)

        input_ids = inputs['input_ids'][idx]
        word_positions = []

        current_input_ids = input_ids.cpu().tolist()
        
        for i in range(len(current_input_ids) - len(word_ids) + 1):
            if current_input_ids[i:i+len(word_ids)] == word_ids:
                word_positions.extend(range(i, i+len(word_ids)))

        if word_positions:
            word_embedding = outputs.last_hidden_state[idx, word_positions, :].mean(dim=0).cpu().numpy()
        else:
            word_embedding = outputs.last_hidden_state[idx].mean(dim=0).cpu().numpy()

        word_embeddings.append(word_embedding)
        valid_sentences.append(sentence)
        valid_words.append(word)

    print(f"Output valid sentences: {len(valid_sentences)}")
    return word_embeddings, valid_sentences, valid_words

def cluster_embeddings(embeddings, true_labels):
    """Cluster embeddings based on number of unique non-zero labels"""
    unique_labels = np.unique(true_labels[true_labels != 0])
    n_clusters = len(unique_labels)
    print(f"Clustering into {n_clusters} clusters")
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = clusterer.fit_predict(embeddings)
    
    # Remap cluster labels to match true labels
    remapped_labels = remap_cluster_labels(true_labels, predicted_labels)
    
    return remapped_labels, clusterer

def plot_clusters(result, model_name, word):
    fig = go.Figure()

    # Get unique non-zero labels
    unique_labels = np.unique(result['true_labels'][result['true_labels'] != 0])
    colors = get_color_palette(len(unique_labels))
    
    # Create precision text for title
    precision_text = "<br>Per-class precision: " + ", ".join(
        f"Class {label}: {result['metrics']['per_class_precision'].get(label, 0):.3f}"
        for label in unique_labels
    )
    
    # Plot points
    for i, label in enumerate(unique_labels):
        # Get precision for this class
        class_precision = result['metrics']['per_class_precision'].get(int(label), 0)
        
        # Plot true labels
        mask = result['true_labels'] == label
        fig.add_trace(go.Scatter(
            x=result['reduced_embeddings'][mask, 0],
            y=result['reduced_embeddings'][mask, 1],
            mode='markers',
            marker=dict(color=colors[i], size=8, symbol='circle'),
            name=f'True Class {label} (Precision: {class_precision:.3f})',
            text=np.array(result['valid_sentences'])[mask],
            hoverinfo='text'
        ))
        
        # Plot predicted clusters
        mask_pred = result['cluster_labels'] == label
        fig.add_trace(go.Scatter(
            x=result['reduced_embeddings'][mask_pred, 0],
            y=result['reduced_embeddings'][mask_pred, 1],
            mode='markers',
            marker=dict(color=colors[i], size=12, symbol='x'),
            name=f'Predicted Class {label}',
            text=np.array(result['valid_sentences'])[mask_pred],
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f'Word Embedding Clusters for "{word}" - {model_name}<br>'
              f'Accuracy: {result["metrics"]["accuracy"]:.3f}, '
              f'Overall Precision: {result["metrics"]["overall_precision"]:.3f}'
              f'{precision_text}',
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Classes",
        height=800,
        width=1000
    )

    return fig

def process_word(word, labeled_df, unlabeled_df, model_name):
    # Process labeled data
    word_df_labeled = labeled_df[labeled_df['word'] == word]
    labeled_sentences = word_df_labeled['sentence'].tolist()
    labeled_words = word_df_labeled['word'].tolist()
    true_labels = word_df_labeled['sense'].to_numpy()

    # Process unlabeled data
    word_df_unlabeled = unlabeled_df[unlabeled_df['word'] == word]
    # Remove sentences that are in labeled dataset
    word_df_unlabeled = word_df_unlabeled[~word_df_unlabeled['sentence'].isin(labeled_sentences)]
    # Remove duplicates
    word_df_unlabeled = word_df_unlabeled.drop_duplicates(subset=['sentence'])
    
    unlabeled_sentences = word_df_unlabeled['sentence'].tolist()
    unlabeled_words = word_df_unlabeled['word'].tolist()

    print(f"Processing {word} with {model_name}:")
    print(f"Labeled sentences: {len(labeled_sentences)}")
    print(f"Unlabeled sentences: {len(unlabeled_sentences)}")

    # Combine sentences for embedding
    all_sentences = labeled_sentences + unlabeled_sentences
    all_words = labeled_words + unlabeled_words

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Get embeddings for all sentences
    word_embeddings, valid_sentences, valid_words = get_word_embeddings(all_sentences, all_words, tokenizer, model)

    # Split embeddings back into labeled and unlabeled
    n_labeled = len(labeled_sentences)
    labeled_embeddings = np.array(word_embeddings[:n_labeled])
    unlabeled_embeddings = np.array(word_embeddings[n_labeled:])
    all_embeddings = np.array(word_embeddings)

    # Cluster all embeddings together
    predicted_labels, clusterer = cluster_embeddings(all_embeddings, np.concatenate([true_labels, np.zeros(len(unlabeled_sentences))]))
    
    # Split predictions back into labeled and unlabeled
    labeled_predictions = predicted_labels[:n_labeled]
    unlabeled_predictions = predicted_labels[n_labeled:]

    # Calculate metrics only on labeled data
    metrics = calculate_metrics(true_labels, labeled_predictions)
    
    # Reduce dimensionality for visualization
    reducer = UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    result = {
        'embeddings': all_embeddings,
        'cluster_labels': predicted_labels,
        'true_labels': true_labels,
        'valid_sentences': valid_sentences,
        'valid_words': valid_words,
        'reduced_embeddings': reduced_embeddings,
        'metrics': metrics
    }

    # Create directory for this word and model
    word_dir = os.path.join('results', word)
    model_dir = os.path.join(word_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save results
    np.save(os.path.join(model_dir, 'embeddings.npy'), all_embeddings)
    np.save(os.path.join(model_dir, 'predicted_labels.npy'), predicted_labels)
    np.save(os.path.join(model_dir, 'true_labels.npy'), true_labels)
    np.save(os.path.join(model_dir, 'reduced_embeddings.npy'), reduced_embeddings)
    
    # Save sentences with their corresponding indices and labels
    with open(os.path.join(model_dir, 'sentences_with_labels.txt'), 'w', encoding='utf-8') as f:
        for idx, (sentence, true_label, pred_label) in enumerate(zip(valid_sentences, true_labels, predicted_labels)):
            f.write(f"{idx}\t{sentence}\t{true_label}\t{pred_label}\n")

    # Generate and save cluster plot
    cluster_plot = plot_clusters(result, model_name, word)
    pio.write_html(cluster_plot, file=os.path.join(model_dir, 'cluster_plot.html'))

    del model, tokenizer
    gc.collect()

    return result

def generate_word_report(word, results, output_file):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Word Analysis Report: {word}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .model-section {{ margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
            .metrics {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            .precision-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            .precision-table th, .precision-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .precision-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Word Analysis Report: {word}</h1>
    """

    for model_name, result in results.items():
        metrics = result['metrics']
        html_content += f"""
        <div class="model-section">
            <h2>{model_name}</h2>
            <div class="metrics">
                <p><strong>Overall Accuracy:</strong> {metrics['accuracy']:.3f}</p>
                <p><strong>Overall Precision:</strong> {metrics['overall_precision']:.3f}</p>
                
                <h3>Per-Class Precision</h3>
                <table class="precision-table">
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                    </tr>
        """
        
        for class_label, precision in metrics['per_class_precision'].items():
            html_content += f"""
                    <tr>
                        <td>Class {class_label}</td>
                        <td>{precision:.3f}</td>
                    </tr>
            """
            
        html_content += """
                </table>
            </div>
            <p><a href="{model_name}/cluster_plot.html" target="_blank">View Cluster Plot</a></p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

def process_all_words(labeled_df, unlabeled_df, model_names):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    unique_words = labeled_df['word'].unique()
    results = {}
    
    for word in unique_words:
        word_results = {}
        print(f"\nProcessing word: {word}")
        
        word_dir = os.path.join(results_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        
        for model_name in model_names:
            result = process_word(word, labeled_df, unlabeled_df, model_name)
            word_results[model_name] = result
        
        output_file = os.path.join(word_dir, f"word_analysis.html")
        generate_word_report(word, word_results, output_file)
        
        results[word] = word_results

    return results

if __name__ == "__main__":
    labeled_df = pd.read_csv('subsetmax.csv')
    unlabeled_df = pd.read_csv('unlabeled_data.csv')  # Add your unlabeled dataset
    
    model_names = [
        'l3cube-pune/kannada-bert',
        'google/muril-base-cased',
        'pierluigic/xl-lexeme'
    ]

    results = process_all_words(labeled_df, unlabeled_df, model_names)
    print("Word analysis complete. Results are available in individual word directories.")