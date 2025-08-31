import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

def plot_training_history(history, model_name='Model', save_dir=None):
    """Plot training and validation accuracy and loss for PyTorch training history."""
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_acc']) + 1)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title(f'{model_name} - Model Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add best accuracy annotations
    best_train_acc = max(history['train_acc'])
    best_val_acc = max(history['val_acc'])
    best_train_epoch = history['train_acc'].index(best_train_acc) + 1
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    
    plt.annotate(f'Best: {best_train_acc:.2f}%', 
                xy=(best_train_epoch, best_train_acc), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'Best: {best_val_acc:.2f}%', 
                xy=(best_val_epoch, best_val_acc), 
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'{model_name} - Model Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add best loss annotations
    best_train_loss = min(history['train_loss'])
    best_val_loss = min(history['val_loss'])
    best_train_loss_epoch = history['train_loss'].index(best_train_loss) + 1
    best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1
    
    plt.annotate(f'Best: {best_train_loss:.4f}', 
                xy=(best_train_loss_epoch, best_train_loss), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'Best: {best_val_loss:.4f}', 
                xy=(best_val_loss_epoch, best_val_loss), 
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name.lower()}_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, class_names, model_name='Model', save_dir=None):
    """Plot confusion matrix with enhanced visualization."""
    
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages for better readability
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations that show both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentage[i, j]
            row.append(f'{count}\n({percentage:.1f}%)')
        annotations.append(row)
    
    # Create heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'},
                square=True, linewidths=0.5)
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name.lower()}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_class_distribution(y_data, class_names, save_dir=None):
    """Plot class distribution in the dataset."""
    
    plt.figure(figsize=(12, 6))
    
    # Count occurrences of each class
    unique, counts = np.unique(y_data, return_counts=True)
    
    # Create subplot for counts
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(unique)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Class Distribution (Counts)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.annotate(f'{count}', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold')
    
    # Create subplot for percentages
    plt.subplot(1, 2, 2)
    percentages = counts / counts.sum() * 100
    bars = plt.bar(range(len(unique)), percentages, color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('Class Distribution (Percentages)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars, percentages):
        plt.annotate(f'{percentage:.1f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Class distribution plot saved to: {save_path}")
    
    plt.show()

def plot_random_search_results(results_path, model_name='Model', save_dir=None):
    """Plot random search optimization results."""
    
    try:
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        all_results = results['all_results']
        best_score = results.get('best_cv_score', results.get('best_score', 0))
        
        if not all_results:
            print("No results found in the file.")
            return
        
        # Extract scores and iterations
        iterations = [r['iteration'] for r in all_results if r.get('cv_score', 0) > 0]
        scores = [r['cv_score'] for r in all_results if r.get('cv_score', 0) > 0]
        
        if not scores:
            print("No valid scores found in the results.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Score progression
        plt.subplot(2, 2, 1)
        plt.plot(iterations, scores, 'bo-', linewidth=2, markersize=8, alpha=0.7)
        plt.axhline(y=best_score, color='r', linestyle='--', linewidth=2, 
                   label=f'Best Score: {best_score:.4f}')
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Cross-Validation Score', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Random Search Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        plt.subplot(2, 2, 2)
        plt.hist(scores, bins=min(20, len(scores)), alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=best_score, color='r', linestyle='--', linewidth=2, 
                   label=f'Best Score: {best_score:.4f}')
        plt.xlabel('Cross-Validation Score', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Score Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Parameter analysis (if we have parameter data)
        plt.subplot(2, 2, 3)
        if len(all_results) > 0 and 'parameters' in all_results[0]:
            # Analyze most important parameter (first one)
            param_names = list(all_results[0]['parameters'].keys())
            if param_names:
                first_param = param_names[0]
                param_values = [r['parameters'][first_param] for r in all_results if r.get('cv_score', 0) > 0]
                
                # Create scatter plot
                plt.scatter(param_values, scores, alpha=0.7, s=100, c='green')
                plt.xlabel(f'{first_param}', fontsize=12, fontweight='bold')
                plt.ylabel('CV Score', fontsize=12, fontweight='bold')
                plt.title(f'Parameter Analysis: {first_param}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
        
        # Plot 4: Best vs Worst parameters comparison
        plt.subplot(2, 2, 4)
        if len(scores) > 1:
            top_5_indices = np.argsort(scores)[-5:]  # Top 5 scores
            top_5_scores = [scores[i] for i in top_5_indices]
            top_5_iterations = [iterations[i] for i in top_5_indices]
            
            plt.bar(range(len(top_5_scores)), top_5_scores, 
                   color=['gold', 'silver', 'chocolate', 'lightcoral', 'lightblue'][:len(top_5_scores)])
            plt.xlabel('Top Iterations', fontsize=12, fontweight='bold')
            plt.ylabel('CV Score', fontsize=12, fontweight='bold')
            plt.title(f'{model_name} - Top 5 Results', fontsize=14, fontweight='bold')
            plt.xticks(range(len(top_5_scores)), 
                      [f'Iter {it}' for it in top_5_iterations], rotation=45)
            
            # Add value labels on bars
            for i, score in enumerate(top_5_scores):
                plt.annotate(f'{score:.4f}', 
                           xy=(i, score), xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{model_name.lower()}_random_search_results.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Random search results plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\n📈 Random Search Summary for {model_name}:")
        print(f"   Total iterations: {len(all_results)}")
        print(f"   Successful iterations: {len(scores)}")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Mean score: {np.mean(scores):.4f}")
        print(f"   Std score: {np.std(scores):.4f}")
        print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
        
    except Exception as e:
        print(f"Error plotting random search results: {e}")
        print("Please check that the results file exists and has the correct format.")
    
    # Extract scores and iterations
    iterations = [r['iteration'] for r in all_results]
    scores = [r['mean_cv_score'] for r in all_results]
    std_scores = [r['std_cv_score'] for r in all_results]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Score progression
    plt.subplot(2, 2, 1)
    plt.errorbar(iterations, scores, yerr=std_scores, marker='o', capsize=5)
    plt.axhline(y=best_score, color='r', linestyle='--', 
                label=f'Best Score: {best_score:.4f}')
    plt.title(f'{model_name} - Random Search Score Progression')
    plt.xlabel('Iteration')
    plt.ylabel('CV Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Score distribution
    plt.subplot(2, 2, 2)
    plt.hist(scores, bins=max(5, len(scores)//3), alpha=0.7, edgecolor='black')
    plt.axvline(x=best_score, color='r', linestyle='--', 
                label=f'Best Score: {best_score:.4f}')
    plt.title(f'{model_name} - Score Distribution')
    plt.xlabel('CV Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 3: Best parameters visualization (for numeric parameters)
    plt.subplot(2, 2, 3)
    best_params = results['best_parameters']
    numeric_params = {k: v for k, v in best_params.items() 
                     if isinstance(v, (int, float))}
    
    if numeric_params:
        param_names = list(numeric_params.keys())
        param_values = list(numeric_params.values())
        
        plt.bar(param_names, param_values)
        plt.title(f'{model_name} - Best Hyperparameters')
        plt.xticks(rotation=45)
        plt.ylabel('Parameter Value')
    else:
        plt.text(0.5, 0.5, 'No numeric parameters\nto visualize', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name} - Best Hyperparameters')
    
    # Plot 4: Parameter importance (correlation with scores)
    plt.subplot(2, 2, 4)
    
    # Analyze parameter importance
    param_importance = analyze_parameter_importance(all_results)
    
    if param_importance:
        params, importance = zip(*param_importance.items())
        plt.bar(params, importance)
        plt.title(f'{model_name} - Parameter Importance')
        plt.xticks(rotation=45)
        plt.ylabel('Correlation with Score')
    else:
        plt.text(0.5, 0.5, 'Parameter importance\nanalysis not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name} - Parameter Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nRandom Search Summary for {model_name}:")
    print(f"{'='*50}")
    print(f"Total iterations: {len(all_results)}")
    print(f"Best CV score: {best_score:.4f}")
    print(f"Average CV score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Best parameters: {best_params}")

def analyze_parameter_importance(all_results):
    """Analyze correlation between parameters and scores."""
    import pandas as pd
    
    # Create DataFrame from results
    data = []
    for result in all_results:
        row = result['parameters'].copy()
        row['score'] = result['mean_cv_score']
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate correlations for numeric parameters only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlations = df[numeric_cols].corr()['score'].drop('score')
        return correlations.to_dict()
    
    return {}

def plot_hyperparameter_heatmap(results_path, model_name='Model'):
    """Plot heatmap of hyperparameter combinations and their scores."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    all_results = results['all_results']
    
    # Create parameter-score matrix for categorical parameters
    import pandas as pd
    
    data = []
    for result in all_results:
        row = result['parameters'].copy()
        row['score'] = result['mean_cv_score']
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Find categorical parameters
    categorical_params = []
    for col in df.columns:
        if col != 'score' and not pd.api.types.is_numeric_dtype(df[col]):
            categorical_params.append(col)
    
    if len(categorical_params) >= 2:
        # Create heatmap for first two categorical parameters
        param1, param2 = categorical_params[:2]
        pivot_table = df.pivot_table(values='score', index=param1, 
                                   columns=param2, aggfunc='mean')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'{model_name} - Hyperparameter Heatmap ({param1} vs {param2})')
        plt.show()
    else:
        print(f"Not enough categorical parameters for heatmap visualization")

def compare_models_random_search(vgg16_results_path, mobilevit_results_path):
    """Compare random search results between VGG-16 and MobileViT models."""
    
    # Load both results
    with open(vgg16_results_path, 'r') as f:
        vgg16_results = json.load(f)
    
    with open(mobilevit_results_path, 'r') as f:
        mobilevit_results = json.load(f)
    
    # Extract scores
    vgg16_scores = [r['mean_cv_score'] for r in vgg16_results['all_results']]
    mobilevit_scores = [r['mean_cv_score'] for r in mobilevit_results['all_results']]
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Score comparison
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(vgg16_scores)+1), vgg16_scores, 
             'o-', label='VGG-16', alpha=0.7)
    plt.plot(range(1, len(mobilevit_scores)+1), mobilevit_scores, 
             's-', label='MobileViT', alpha=0.7)
    plt.title('Random Search Score Progression Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('CV Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Score distribution comparison
    plt.subplot(2, 2, 2)
    plt.hist(vgg16_scores, alpha=0.6, label='VGG-16', bins=10)
    plt.hist(mobilevit_scores, alpha=0.6, label='MobileViT', bins=10)
    plt.title('Score Distribution Comparison')
    plt.xlabel('CV Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 3: Box plot comparison
    plt.subplot(2, 2, 3)
    plt.boxplot([vgg16_scores, mobilevit_scores], 
                labels=['VGG-16', 'MobileViT'])
    plt.title('Score Distribution Box Plot')
    plt.ylabel('CV Score')
    
    # Plot 4: Best scores comparison
    plt.subplot(2, 2, 4)
    models = ['VGG-16', 'MobileViT']
    best_scores = [vgg16_results['best_score'], mobilevit_results['best_score']]
    
    bars = plt.bar(models, best_scores, color=['blue', 'orange'], alpha=0.7)
    plt.title('Best Scores Comparison')
    plt.ylabel('Best CV Score')
    
    # Add value labels on bars
    for bar, score in zip(bars, best_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print("="*50)
    print(f"VGG-16 Best Score: {vgg16_results['best_score']:.4f}")
    print(f"MobileViT Best Score: {mobilevit_results['best_score']:.4f}")
    print(f"VGG-16 Average Score: {np.mean(vgg16_scores):.4f} ± {np.std(vgg16_scores):.4f}")
    print(f"MobileViT Average Score: {np.mean(mobilevit_scores):.4f} ± {np.std(mobilevit_scores):.4f}")
    
    winner = "VGG-16" if vgg16_results['best_score'] > mobilevit_results['best_score'] else "MobileViT"
    print(f"Best performing model: {winner}")