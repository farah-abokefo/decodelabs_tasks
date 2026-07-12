"""
PROJECT 2: DATA CLASSIFICATION USING AI
DecodeLabs - Batch 2026
Supervised Learning with K-Nearest Neighbors (KNN)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. DATA LOADING & UNDERSTANDING
# ============================================

def load_and_explore_data():
    """
    PHASE 1: Load and understand the dataset
    Following the IPO Framework
    """
    print("="*60)
    print("📊 PHASE 1: DATA LOADING & EXPLORATION")
    print("="*60)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Labels: 0=Setosa, 1=Versicolor, 2=Virginica
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y
    df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    
    print(f"\n📌 Dataset Overview:")
    print(f"   • Samples: {X.shape[0]} (Balanced)")
    print(f"   • Classes: {len(iris.target_names)}")
    print(f"   • Features: {X.shape[1]} dimensions")
    print(f"   • Class distribution:")
    print(df['species_name'].value_counts().to_string())
    
    print(f"\n📌 Feature Statistics:")
    print(df.describe().round(2))
    
    print(f"\n📌 First 5 samples:")
    print(df.head())
    
    return X, y, df, iris

# ============================================
# 2. DATA PREPROCESSING
# ============================================

def preprocess_data(X, y, test_size=0.3, random_state=42):
    """
    PHASE 2: Data Preprocessing
    - Train-Test Split (70-30)
    - Feature Scaling (StandardScaler)
    """
    print("\n" + "="*60)
    print("🔧 PHASE 2: DATA PREPROCESSING")
    print("="*60)
    
    # Step 1: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"\n📌 Train-Test Split:")
    print(f"   • Training samples: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    print(f"   • Test samples: {X_test.shape[0]} ({test_size*100:.0f}%)")
    
    # Step 2: Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📌 Feature Scaling Applied:")
    print(f"   • Mean ≈ 0, Variance ≈ 1")
    print(f"   • Training data scaled: {X_train_scaled.shape}")
    print(f"   • Test data scaled: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================
# 3. MODEL TRAINING (KNN)
# ============================================

def train_knn_model(X_train, y_train, k=5):
    """
    PHASE 3: Model Training
    Using K-Nearest Neighbors algorithm
    """
    print("\n" + "="*60)
    print("🤖 PHASE 3: MODEL TRAINING")
    print("="*60)
    
    # Instantiate the model
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    
    # Fit the model (memorize the map)
    model.fit(X_train, y_train)
    
    print(f"\n📌 Model Details:")
    print(f"   • Algorithm: K-Nearest Neighbors")
    print(f"   • K value: {k}")
    print(f"   • Distance metric: Euclidean")
    print(f"   • Training complete! ✅")
    
    return model

# ============================================
# 4. MODEL EVALUATION
# ============================================

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    PHASE 4: Model Evaluation
    - Accuracy
    - Confusion Matrix
    - Classification Report (Precision, Recall, F1-Score)
    - Cross-Validation
    """
    print("\n" + "="*60)
    print("📈 PHASE 4: MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # 1. Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📌 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📌 Confusion Matrix:")
    print(cm)
    
    # 3. Classification Report
    print(f"\n📌 Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    # 4. F1 Score (Micro and Macro)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"\n📌 F1 Scores:")
    print(f"   • Micro F1: {f1_micro:.4f}")
    print(f"   • Macro F1: {f1_macro:.4f}")
    
    # 5. Cross-Validation (5-fold)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n📌 Cross-Validation (5-fold):")
    print(f"   • Mean CV Score: {cv_scores.mean():.4f}")
    print(f"   • Std CV Score: {cv_scores.std():.4f}")
    print(f"   • Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    return y_pred, cm, accuracy

# ============================================
# 5. VISUALIZATION
# ============================================

def create_visualizations(X_train, X_test, y_train, y_test, y_pred, cm, model, iris, scaler):
    """
    PHASE 5: Visualizations
    - Confusion Matrix
    - Feature Distributions
    - Decision Boundary (2D projection)
    """
    print("\n" + "="*60)
    print("🎨 PHASE 5: VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12), dpi=100)
    
    # Use GridSpec for better control over subplot sizes
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                ax=ax1, annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.tick_params(labelsize=11)
    
    # 2. Feature Distribution (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    df_train = pd.DataFrame(X_train[:, :2], columns=iris.feature_names[:2])
    df_train['species'] = y_train
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    df_train['species_name'] = df_train['species'].map(species_map)
    
    colors = {'Setosa': '#2ecc71', 'Versicolor': '#3498db', 'Virginica': '#e74c3c'}
    markers = {'Setosa': 'o', 'Versicolor': 's', 'Virginica': 'D'}
    
    for species in ['Setosa', 'Versicolor', 'Virginica']:
        subset = df_train[df_train['species_name'] == species]
        ax2.scatter(subset[iris.feature_names[0]], 
                   subset[iris.feature_names[1]], 
                   label=species, 
                   alpha=0.7, 
                   s=60,
                   color=colors[species],
                   marker=markers[species],
                   edgecolors='white',
                   linewidth=1)
    ax2.set_xlabel(iris.feature_names[0], fontsize=12)
    ax2.set_ylabel(iris.feature_names[1], fontsize=12)
    ax2.set_title('Training Data Distribution (2D)', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # 3. Model Performance Metrics (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ['Accuracy', 'F1 (Micro)', 'F1 (Macro)']
    values = [
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average='micro'),
        f1_score(y_test, y_pred, average='macro')
    ]
    colors_bars = ['#27ae60', '#2980b9', '#f39c12']
    bars = ax3.bar(metrics, values, color=colors_bars, width=0.6, edgecolor='black', linewidth=1)
    ax3.set_ylim(0, 1.15)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=11)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 4. Prediction Results Distribution (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']
    
    correct_count = results_df['Correct'].sum()
    incorrect_count = len(results_df) - correct_count
    
    # Create better visualization with colored bars
    x_pos = range(len(results_df))
    colors_pred = ['#2ecc71' if c else '#e74c3c' for c in results_df['Correct']]
    
    ax4.bar(x_pos, [1]*len(results_df), color=colors_pred, alpha=0.8, width=0.8)
    ax4.set_title(f'Prediction Results\n(✅ {correct_count} Correct, ❌ {incorrect_count} Incorrect)',
                 fontsize=14, fontweight='bold', pad=15)
    ax4.set_xlabel('Test Sample Index', fontsize=12)
    ax4.set_ylabel('Prediction Status', fontsize=12)
    ax4.set_yticks([])
    ax4.set_xlim(-0.5, len(results_df) - 0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.tick_params(labelsize=11)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label=f'Correct ({correct_count})', edgecolor='black'),
                      Patch(facecolor='#e74c3c', label=f'Incorrect ({incorrect_count})', edgecolor='black')]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Adjust layout with proper spacing
    plt.tight_layout(pad=3.0)
    
    # Save with high quality
    plt.savefig('outputs/classification_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📌 Visualization saved: outputs/classification_results.png")
    
    # Display with proper window size
    plt.show()
    
    return fig

# ============================================
# 6. K VALUE TUNING
# ============================================

def tune_k_value(X_train, y_train, X_test, y_test, max_k=20):
    """
    OPTIONAL: Tuning the K value
    Finding the optimal number of neighbors
    """
    print("\n" + "="*60)
    print("🎯 K VALUE TUNING (Hyperparameter Optimization)")
    print("="*60)
    
    k_values = range(1, max_k + 1)
    train_accuracies = []
    test_accuracies = []
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # Plot results with better sizing
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Create the plot with better styling
    plt.plot(k_values, train_accuracies, label='Training Accuracy', 
             marker='o', color='#3498db', linewidth=2.5, markersize=8)
    plt.plot(k_values, test_accuracies, label='Testing Accuracy', 
             marker='s', color='#e74c3c', linewidth=2.5, markersize=8)
    
    plt.xlabel('K Value (Number of Neighbors)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=14, fontweight='bold')
    plt.title('K Value Tuning: Finding the Optimal Number of Neighbors', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(range(1, max_k + 1), fontsize=11)
    plt.yticks(fontsize=11)
    
    # Highlight optimal K with a vertical line
    optimal_k = k_values[np.argmax(test_accuracies)]
    max_acc = max(test_accuracies)
    
    # Add vertical line for optimal K
    plt.axvline(x=optimal_k, color='green', linestyle='--', alpha=0.7, 
                linewidth=2, label=f'Optimal K = {optimal_k}')
    
    # Highlight the optimal point
    plt.scatter([optimal_k], [max_acc], color='red', s=150, zorder=5, 
                edgecolors='black', linewidth=2)
    plt.annotate(f'K = {optimal_k}\nAcc = {max_acc:.3f}', 
                xy=(optimal_k, max_acc), 
                xytext=(optimal_k + 1, max_acc - 0.05),
                fontsize=12,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Add shaded region for optimal range
    plt.axvspan(optimal_k - 0.5, optimal_k + 0.5, alpha=0.1, color='green')
    
    plt.legend(fontsize=11, loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig('outputs/k_value_tuning.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📌 K-value tuning plot saved: outputs/k_value_tuning.png")
    plt.show()
    
    # Find optimal K
    print(f"\n📌 Optimal K value: {optimal_k}")
    print(f"   • Best test accuracy: {max(test_accuracies):.4f}")
    print(f"   • Training accuracy at K={optimal_k}: {train_accuracies[optimal_k-1]:.4f}")
    
    return optimal_k, test_accuracies

# ============================================
# 7. MODEL COMPARISON (Optional)
# ============================================

def compare_algorithms(X_train, X_test, y_train, y_test):
    """
    Compare KNN with other algorithms
    """
    print("\n" + "="*60)
    print(" ALGORITHM COMPARISON")
    print("="*60)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    models = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=200)
    }
    
    results = {}
    print(f"\n Algorithm Performance:")
    print("-" * 50)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = {'accuracy': acc, 'f1': f1}
        print(f"   • {name:<20}: Acc: {acc:.4f} ({acc*100:.2f}%) | F1: {f1:.4f}")
    print("-" * 50)
    
    # Plot comparison with better sizing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=100)
    
    names = list(results.keys())
    acc_values = [results[name]['accuracy'] for name in names]
    f1_values = [results[name]['f1'] for name in names]
    
    # Accuracy plot
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, acc_values, width, label='Accuracy', 
                    color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, f1_values, width, label='F1 (Macro)', 
                    color='#e67e22', edgecolor='black', linewidth=1.5)
    
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Create a heatmap style comparison (second subplot)
    import pandas as pd
    comparison_df = pd.DataFrame({
        'Algorithm': names,
        'Accuracy': acc_values,
        'F1 Score': f1_values
    })
    
    # Display as table
    ax2.axis('tight')
    ax2.axis('off')
    table_data = comparison_df.values.tolist()
    table_data.insert(0, ['Algorithm', 'Accuracy', 'F1 Score'])
    
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Color coding for table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                if j == 0:  # Algorithm name
                    cell.set_facecolor('#ecf0f1')
                else:
                    value = table_data[i][j]
                    if isinstance(value, float):
                        if value >= 0.95:
                            cell.set_facecolor('#2ecc71')
                        elif value >= 0.85:
                            cell.set_facecolor('#f1c40f')
                        else:
                            cell.set_facecolor('#e74c3c')
    
    plt.suptitle('Algorithm Comparison on Iris Dataset', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/algorithm_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📌 Algorithm comparison saved: outputs/algorithm_comparison.png")
    plt.show()
    
    return results

# ============================================
# 8. MAIN EXECUTION
# ============================================

def main():
    """
    Main execution following the full architecture
    """
    print("\n" + "="*60)
    print("🤖 PROJECT 2: DATA CLASSIFICATION USING AI")
    print("DecodeLabs - Batch 2026")
    print("Supervised Learning with K-Nearest Neighbors")
    print("="*60)
    
    # Create outputs directory
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("\n📁 Created 'outputs' directory for saving visualizations")
    
    # Set matplotlib parameters for better display
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    
    # PHASE 1: Load and explore data
    X, y, df, iris = load_and_explore_data()
    
    # PHASE 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.3)
    
    # PHASE 3: Train model (with K=5 as default)
    model = train_knn_model(X_train, y_train, k=5)
    
    # PHASE 4: Evaluate model
    y_pred, cm, accuracy = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # PHASE 5: Visualizations
    create_visualizations(X_train, X_test, y_train, y_test, y_pred, cm, model, iris, scaler)
    
    # OPTIONAL: K Value Tuning
    optimal_k, _ = tune_k_value(X_train, y_train, X_test, y_test, max_k=15)
    
    # OPTIONAL: Algorithm Comparison
    compare_algorithms(X_train, X_test, y_train, y_test)
    
    # Final Summary
    print("\n" + "="*60)
    print("✅ PROJECT 2 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\n📊 Final Model Performance:")
    print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Model: KNN (k={model.n_neighbors})")
    print(f"   • Optimal K found: {optimal_k}")
    print(f"   • Test samples: {len(y_test)}")
    print(f"\n🏆 You've mastered supervised learning!")
    print("🚀 Ready for Project 3: Deep Learning & CNNs?")
    print("="*60)

# ============================================
# 9. ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution interrupted by user.")
        print("🔄 You can run the script again anytime.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("🔄 Please check your data and try again.")