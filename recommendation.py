"""
PROJECT 3: AI RECOMMENDATION LOGIC
DecodeLabs - Batch 2026
Tech Stack Recommender using Content-Based Filtering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. DATA CREATION & LOADING
# ============================================

def create_dataset():
    """
    Create the Tech Stack dataset with job roles and required skills
    Following the IPO Framework - RAW MATERIAL
    """
    print("="*60)
    print(" PHASE 1: DATA LOADING & EXPLORATION")
    print("="*60)
    
    # Dataset: Job roles mapped to required skills
    data = {
        'job_role': [
            'Data Scientist',
            'Machine Learning Engineer',
            'Cloud Architect',
            'DevOps Engineer',
            'Full Stack Developer',
            'Backend Developer',
            'Frontend Developer',
            'Data Engineer',
            'AI Researcher',
            'Software Engineer',
            'Database Administrator',
            'Security Engineer',
            'Network Engineer',
            'Mobile Developer',
            'Game Developer'
        ],
        'skills': [
            'Python SQL Machine Learning Statistics Data Visualization Deep Learning',
            'Python TensorFlow PyTorch Docker Kubernetes AWS GCP',
            'AWS Azure GCP Terraform Kubernetes Docker Python Networking',
            'Docker Kubernetes Jenkins AWS Linux Python Bash CI/CD',
            'JavaScript React Node.js MongoDB Express HTML CSS Git',
            'Python Java Spring Boot REST APIs SQL Docker AWS',
            'JavaScript React HTML CSS TypeScript Webpack Git',
            'Python SQL Spark Hadoop AWS GCP Data Warehousing ETL',
            'Python PyTorch TensorFlow Deep Learning NLP Computer Vision Research',
            'Java Python C++ Git SQL Algorithms Data Structures',
            'SQL Oracle MySQL PostgreSQL Database Design Performance Tuning',
            'Python Cybersecurity Network Security Linux Cryptography Firewalls',
            'Python Networking Cisco AWS Routing Switching Linux',
            'Java Kotlin Swift React Native Android iOS Firebase',
            'C++ Unity Unreal Engine 3D Modeling Game Design Physics'
        ]
    }
    
    df = pd.DataFrame(data)
    
    print(f"\n Dataset Overview:")
    print(f"   • Total job roles: {len(df)}")
    print(f"   • Skills per role: {df['skills'].str.split().str.len().mean():.1f} (average)")
    print(f"   • Total unique skills: {len(set(' '.join(df['skills']).lower().split()))}")
    
    print(f"\n Sample Job Roles:")
    for i, row in df.head(5).iterrows():
        print(f"   • {row['job_role']}: {row['skills'][:50]}...")
    
    return df

# ============================================
# 2. FEATURE EXTRACTION (TF-IDF)
# ============================================

def extract_features(df):
    """
    PHASE 2: Feature Extraction using TF-IDF
    Upgrading from binary vectors to TF-IDF weighting
    """
    print("\n" + "="*60)
    print(" PHASE 2: FEATURE EXTRACTION (TF-IDF)")
    print("="*60)
    
    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=100,
        min_df=1,
        max_df=0.95
    )
    
    # Fit and transform the skills column
    tfidf_matrix = vectorizer.fit_transform(df['skills'])
    
    # Convert to array for easier viewing
    tfidf_array = tfidf_matrix.toarray()
    
    # Get feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n TF-IDF Extraction Results:")
    print(f"   • Shape: {tfidf_array.shape} (documents × features)")
    print(f"   • Vocabulary size: {len(feature_names)} unique skills")
    print(f"   • Non-zero entries: {tfidf_matrix.nnz}")
    print(f"   • Sparsity: {100 * (1 - tfidf_matrix.nnz / (tfidf_array.shape[0] * tfidf_array.shape[1])):.2f}%")
    
    # Show sample TF-IDF weights
    print(f"\n Sample TF-IDF Weights for 'Data Scientist':")
    sample_weights = tfidf_array[0]
    top_indices = np.argsort(sample_weights)[-5:][::-1]
    for idx in top_indices:
        if sample_weights[idx] > 0:
            print(f"   • {feature_names[idx]}: {sample_weights[idx]:.4f}")
    
    return tfidf_matrix, tfidf_array, vectorizer, feature_names

# ============================================
# 3. USER PROFILE CREATION
# ============================================

def get_user_profile():
    """
    PHASE 3: User Input & Profile Creation
    Following the "Ingestion" step from the pipeline
    """
    print("\n" + "="*60)
    print(" PHASE 3: USER INPUT & PROFILING")
    print("="*60)
    
    print("\n Enter your skills (minimum 3):")
    print("   Tip: Python, JavaScript, SQL, AWS, Docker, etc.")
    print("   Enter skills separated by commas (e.g., Python, SQL, Machine Learning)")
    
    while True:
        user_input = input("\n👤 Your skills: ").strip()
        
        if not user_input:
            print(" Please enter at least one skill.")
            continue
        
        # Split by comma and clean
        skills = [s.strip().lower() for s in user_input.split(',')]
        skills = [s for s in skills if s]  # Remove empty strings
        
        if len(skills) < 3:
            print(f" You entered {len(skills)} skill(s). Please enter at least 3 skills.")
            continue
        
        break
    
    # Join skills back to a single string
    user_skills_text = ' '.join(skills)
    
    print(f"\n Skills captured: {', '.join(skills)}")
    print(f"   • Total skills: {len(skills)}")
    
    return user_skills_text, skills

def create_user_profile(user_skills_text, vectorizer, df):
    """
    Transform user skills into a TF-IDF vector
    """
    print("\n Creating User Profile Vector...")
    
    # Transform user skills using the same vectorizer
    user_profile = vectorizer.transform([user_skills_text])
    user_array = user_profile.toarray()
    
    print(f"   • User profile shape: {user_array.shape}")
    print(f"   • Non-zero features: {user_profile.nnz}")
    
    # Show which skills were found in the vocabulary
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = user_profile.nonzero()[1]
    found_skills = [feature_names[i] for i in nonzero_indices]
    
    if found_skills:
        print(f"   • Skills found in vocabulary: {', '.join(found_skills[:5])}")
    else:
        print(f"   •  No skills matched the vocabulary! Try using more common skill names.")
    
    return user_profile, user_array

# ============================================
# 4. SIMILARITY CALCULATION
# ============================================

def calculate_similarity(user_profile, tfidf_matrix, df):
    """
    PHASE 4: Similarity Calculation
    Using Cosine Similarity - Industry Standard
    """
    print("\n" + "="*60)
    print(" PHASE 4: SIMILARITY CALCULATION")
    print("="*60)
    
    # Calculate cosine similarity between user profile and all job roles
    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    
    # Create results dataframe
    results = df.copy()
    results['similarity_score'] = similarity_scores
    results['similarity_percentage'] = similarity_scores * 100
    
    # Sort by similarity score (descending)
    results = results.sort_values('similarity_score', ascending=False)
    
    print(f"\n Similarity Results:")
    print(f"   • Total job roles evaluated: {len(results)}")
    print(f"   • Highest similarity score: {results.iloc[0]['similarity_score']:.4f}")
    print(f"   • Lowest similarity score: {results.iloc[-1]['similarity_score']:.4f}")
    
    return results

def calculate_jaccard_similarity(user_skills, df):
    """
    OPTIONAL: Jaccard Similarity for comparison
    Simple overlapping tags (binary)
    """
    print("\n📊 Optional: Calculating Jaccard Similarity...")
    
    user_skills_set = set(user_skills)
    scores = []
    
    for skills in df['skills']:
        job_skills_set = set(skills.lower().split())
        intersection = len(user_skills_set.intersection(job_skills_set))
        union = len(user_skills_set.union(job_skills_set))
        
        if union == 0:
            score = 0
        else:
            score = intersection / union
        scores.append(score)
    
    return scores

# ============================================
# 5. TOP-N RECOMMENDATIONS
# ============================================

def get_recommendations(results, n=3):
    """
    PHASE 5: Filtering & Recommendations
    Step 4 of the pipeline - Top-N filtering
    """
    print("\n" + "="*60)
    print(" PHASE 5: TOP-N RECOMMENDATIONS")
    print("="*60)
    
    # Get top N recommendations
    top_n = results.head(n)
    
    print(f"\n📌 Top {n} Recommended Job Roles:")
    print("-" * 50)
    for i, (idx, row) in enumerate(top_n.iterrows(), 1):
        print(f"\n{i}.  {row['job_role']}")
        print(f"   Similarity: {row['similarity_percentage']:.1f}%")
        print(f"   Key Skills: {', '.join(row['skills'].split()[:5])}")
        print(f"   Confidence: {' High' if row['similarity_score'] > 0.5 else ' Medium' if row['similarity_score'] > 0.3 else ' Low'}")
    
    return top_n

# ============================================
# 6. VISUALIZATIONS (FIXED)
# ============================================

def create_visualizations(results, user_skills, top_n, df):
    """
    PHASE 6: Visualizations - IMPROVED VERSION
    Better sizing, layout, and visual quality
    """
    print("\n" + "="*60)
    print(" PHASE 6: VISUALIZATIONS")
    print("="*60)
    
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with proper sizing
    fig = plt.figure(figsize=(18, 12), dpi=120)
    
    # Use GridSpec for better control
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # 1. Similarity Scores Bar Chart (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    top_10 = results.head(10)
    
    # Color gradient based on score
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(top_10)))[::-1]
    
    bars = ax1.barh(top_10['job_role'], top_10['similarity_percentage'], 
                    color=colors, edgecolor='black', linewidth=0.5, height=0.7)
    
    ax1.set_xlabel('Similarity Score (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Top 10 Job Role Matches', fontsize=16, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=10)
    
    # Add value labels with better positioning
    for bar, score in zip(bars, top_10['similarity_percentage']):
        ax1.text(score + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # Add a vertical line at 50% for reference
    ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50% Threshold')
    
    # 2. Skill Match Heatmap (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for heatmap
    top_10_skills = top_10['skills'].str.lower().str.split().tolist()
    user_skills_set = set([s.lower() for s in user_skills])
    
    # Create skill presence matrix
    all_skills = set()
    for skills in top_10_skills:
        all_skills.update(skills)
    all_skills = list(all_skills)
    
    # Create matrix
    skill_matrix = []
    for skills in top_10_skills:
        row = [1 if s in user_skills_set else 0 for s in all_skills]
        skill_matrix.append(row)
    
    if skill_matrix and len(skill_matrix[0]) > 0:
        heatmap_data = np.array(skill_matrix)
        
        # If too many skills, show only top matches
        if heatmap_data.shape[1] > 20:
            skill_freq = np.sum(heatmap_data, axis=0)
            top_skill_indices = np.argsort(skill_freq)[-20:][::-1]
            heatmap_data = heatmap_data[:, top_skill_indices]
            display_skills = [all_skills[i] for i in top_skill_indices]
        else:
            display_skills = all_skills
        
        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=1)
        
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels(top_10['job_role'].tolist(), fontsize=10)
        ax2.set_xticks(range(len(display_skills)))
        
        # Rotate x-labels for better readability
        if len(display_skills) <= 10:
            ax2.set_xticklabels(display_skills, rotation=45, ha='right', fontsize=9)
        else:
            ax2.set_xticklabels(display_skills, rotation=90, ha='center', fontsize=8)
        
        ax2.set_title('Skill Match Heatmap (Top 10 Roles)', fontsize=16, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Skill Match', fontsize=11)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Not Matching', 'Matching'])
    
    # 3. Similarity Distribution (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Histogram with better styling
    n, bins, patches = ax3.hist(results['similarity_percentage'], bins=15, 
                                edgecolor='black', alpha=0.7, color='#3498db',
                                rwidth=0.9)
    
    # Color bins based on value
    for i, patch in enumerate(patches):
        if i < len(patches) // 3:
            patch.set_facecolor('#e74c3c')
        elif i < 2 * len(patches) // 3:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#2ecc71')
    
    # Add vertical lines for top 3
    colors_lines = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['🥇 Best', '🥈 2nd', '🥉 3rd']
    for i in range(3):
        if i < len(results):
            score = results.iloc[i]['similarity_percentage']
            ax3.axvline(x=score, color=colors_lines[i], linestyle='--', 
                       linewidth=2.5, alpha=0.8, label=f'{labels[i]}: {score:.1f}%')
    
    ax3.set_xlabel('Similarity Score (%)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Number of Job Roles', fontsize=13, fontweight='bold')
    ax3.set_title('Similarity Score Distribution', fontsize=16, fontweight='bold', pad=15)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.tick_params(labelsize=10)
    
    # 4. Skill Comparison (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get top 10 job roles and their skill counts
    top_10_roles = results.head(10)
    skill_counts = top_10_roles['skills'].str.split().str.len().tolist()
    
    # Calculate matching skills count
    user_skills_set = set([s.lower() for s in user_skills])
    matching_counts = []
    for skills in top_10_roles['skills'].str.lower().str.split():
        matching_counts.append(len(set(skills).intersection(user_skills_set)))
    
    # Calculate missing skills count
    missing_counts = [skill_counts[i] - matching_counts[i] for i in range(len(skill_counts))]
    
    x = np.arange(len(top_10_roles))
    
    # Stacked bar chart
    ax4.bar(x, matching_counts, width=0.7, label='Matching Skills', 
            color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax4.bar(x, missing_counts, width=0.7, bottom=matching_counts, 
            label='Skills to Learn', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax4.set_xlabel('Job Roles', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Number of Skills', fontsize=13, fontweight='bold')
    ax4.set_title('Skill Breakdown: Matching vs Missing', fontsize=16, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_10_roles['job_role'].tolist(), rotation=20, ha='right', fontsize=10)
    ax4.legend(loc='upper right', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.tick_params(labelsize=10)
    
    # Add value labels on bars
    for i in range(len(top_10_roles)):
        total = skill_counts[i]
        ax4.text(i, total + 0.3, f'Total: {total}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        if matching_counts[i] > 0:
            ax4.text(i, matching_counts[i]/2, f'{matching_counts[i]}', 
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Adjust layout with proper spacing
    plt.tight_layout(pad=3.0)
    
    # Save with high quality
    plt.savefig('outputs/recommendation_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', pad_inches=0.2)
    print(f"\n Visualization saved: outputs/recommendation_results.png")
    
    # Display with proper window size
    plt.show()
    
    return fig

# ============================================
# 7. DETAILED RECOMMENDATION SUMMARY
# ============================================

def display_recommendation_summary(top_n, user_skills, results):
    """
    Display detailed summary of recommendations
    """
    print("\n" + "="*60)
    print(" DETAILED RECOMMENDATION SUMMARY")
    print("="*60)
    
    user_skills_set = set([s.lower() for s in user_skills])
    
    for i, (idx, row) in enumerate(top_n.iterrows(), 1):
        job_skills = set(row['skills'].lower().split())
        matching_skills = user_skills_set.intersection(job_skills)
        missing_skills = job_skills - user_skills_set
        
        print(f"\n #{i}: {row['job_role']}")
        print(f"   ┌─ Similarity Score: {row['similarity_percentage']:.1f}%")
        print(f"   ├─ Matching Skills ({len(matching_skills)}):")
        if matching_skills:
            print(f"   │   {', '.join(list(matching_skills)[:5])}")
            if len(matching_skills) > 5:
                print(f"   │  └─ +{len(matching_skills)-5} more")
        else:
            print("   │  └─ No direct skill matches found")
        
        print(f"   └─ Skills to Learn ({len(missing_skills)}):")
        if missing_skills:
            print(f"       {', '.join(list(missing_skills)[:5])}")
            if len(missing_skills) > 5:
                print(f"      └─ +{len(missing_skills)-5} more")
        else:
            print("       You have all the skills for this role!")
        
        # Learning path recommendation
        if len(missing_skills) <= 2:
            print(f"    You're a great fit! Ready to apply!")
        elif len(missing_skills) <= 4:
            print(f"    Close match! Consider learning the missing skills.")
        else:
            print(f"    Good starting point. Focus on acquiring key missing skills.")

# ============================================
# 8. COLD START HANDLING
# ============================================

def handle_cold_start(user_skills, results, df):
    """
    Handle the Cold Start Problem
    - If no matches found, provide trending/fallback recommendations
    """
    print("\n" + "="*60)
    print(" COLD START HANDLING")
    print("="*60)
    
    # Check if user has any matches
    if results['similarity_score'].max() == 0:
        print("\n No matches found! This is a Cold Start scenario.")
        print(" Using fallback strategy: Trending/Global Popularity")
        
        # Fallback: Recommend most popular roles based on skill coverage
        fallback = df.copy()
        fallback['skill_count'] = fallback['skills'].str.split().str.len()
        fallback = fallback.sort_values('skill_count', ascending=False)
        
        print(f"\n Fallback Recommendations (Based on Role Popularity):")
        for i, row in fallback.head(3).iterrows():
            print(f"\n   • {row['job_role']}")
            print(f"     Skills: {row['skills']}")
            print(f"     Coverage: {row['skill_count']} skills")
        
        return fallback.head(3)
    
    return None

# ============================================
# 9. SIMILARITY HEATMAP (ADDED)
# ============================================

def create_similarity_heatmap(df, user_skills):
    """
    Create a comprehensive similarity heatmap
    Shows similarity between all job roles and user skills
    """
    print("\n" + "="*60)
    print(" CREATING SIMILARITY HEATMAP")
    print("="*60)
    
    # Create TF-IDF matrix for all roles
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['skills'])
    
    # Create user profile
    user_skills_text = ' '.join(user_skills)
    user_profile = vectorizer.transform([user_skills_text])
    
    # Calculate similarity
    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    
    # Create a full similarity matrix for visualization
    full_similarity = cosine_similarity(tfidf_matrix)
    
    # Plot heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    
    # 1. Full similarity heatmap
    im1 = ax1.imshow(full_similarity, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xticks(range(len(df)))
    ax1.set_yticks(range(len(df)))
    ax1.set_xticklabels(df['job_role'], rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(df['job_role'], fontsize=9)
    ax1.set_title('Job Role Similarity Matrix', fontsize=14, fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j and full_similarity[i, j] > 0.1:
                ax1.text(j, i, f'{full_similarity[i, j]:.2f}',
                        ha="center", va="center", 
                        color="black" if full_similarity[i, j] > 0.5 else "white",
                        fontsize=8)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Similarity Score', fontsize=11)
    
    # 2. User similarity bar chart
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_roles = df.iloc[sorted_indices]['job_role'].tolist()
    sorted_scores = similarity_scores[sorted_indices] * 100
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(sorted_roles)))[::-1]
    bars = ax2.barh(sorted_roles, sorted_scores, color=colors, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Similarity Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Your Skills vs Job Roles', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for bar, score in zip(bars, sorted_scores):
        if score > 0:
            ax2.text(score + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{score:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/similarity_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n Similarity heatmap saved: outputs/similarity_heatmap.png")
    plt.show()
    
    return fig

# ============================================
# 10. MAIN EXECUTION
# ============================================

def main():
    """
    Main execution following the full architecture
    """
    print("\n" + "="*60)
    print(" PROJECT 3: AI RECOMMENDATION LOGIC")
    print("DecodeLabs - Batch 2026")
    print("Tech Stack Recommender - Content-Based Filtering")
    print("="*60)
    
    # Create outputs directory
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("\n📁 Created 'outputs' directory for saving visualizations")
    
    # PHASE 1: Load data
    df = create_dataset()
    
    # PHASE 2: Feature Extraction (TF-IDF)
    tfidf_matrix, tfidf_array, vectorizer, feature_names = extract_features(df)
    
    # PHASE 3: User Input & Profile
    user_skills_text, user_skills = get_user_profile()
    user_profile, user_array = create_user_profile(user_skills_text, vectorizer, df)
    
    # PHASE 4: Similarity Calculation
    results = calculate_similarity(user_profile, tfidf_matrix, df)
    
    # Optional: Jaccard Similarity for comparison
    jaccard_scores = calculate_jaccard_similarity(user_skills, df)
    results['jaccard_similarity'] = jaccard_scores
    
    # Cold Start Check
    fallback = handle_cold_start(user_skills, results, df)
    
    # PHASE 5: Top-N Recommendations
    top_n = get_recommendations(results, n=3)
    
    # PHASE 6: Visualizations
    create_visualizations(results, user_skills, top_n, df)
    
    # Additional: Similarity Heatmap
    create_similarity_heatmap(df, user_skills)
    
    # PHASE 7: Detailed Summary
    display_recommendation_summary(top_n, user_skills, results)
    
    # Final Summary
    print("\n" + "="*60)
    print(" PROJECT 3 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\n Recommendation Engine Summary:")
    print(f"   • Skills provided: {len(user_skills)}")
    print(f"   • Job roles evaluated: {len(df)}")
    print(f"   • Top match: {top_n.iloc[0]['job_role']} ({top_n.iloc[0]['similarity_percentage']:.1f}%)")
    print(f"   • Algorithm: Content-Based Filtering + TF-IDF + Cosine Similarity")
    print(f"\n You've built a Digital Matchmaker!")
    print("="*60)

# ============================================
# 11. ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Execution interrupted by user.")
        print(" You can run the script again anytime.")
    except Exception as e:
        print(f"\n An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print(" Please check your data and try again.")