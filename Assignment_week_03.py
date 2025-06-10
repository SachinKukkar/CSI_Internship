# Titanic Dataset Visualization and Exploratory Data Analysis
# Complete Python code for comprehensive data visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the Titanic dataset and perform initial exploration"""
    # Load the dataset (you can download from Kaggle or use seaborn's built-in dataset)
    try:
        # Try to load from seaborn first (built-in dataset)
        titanic = sns.load_dataset('titanic')
        print("✅ Dataset loaded from seaborn")
    except:
        # Alternative: load from CSV if you have the file
        # titanic = pd.read_csv('titanic.csv')
        print("❌ Please ensure you have the Titanic dataset available")
        return None
    
    # Basic dataset information
    print("=" * 50)
    print("TITANIC DATASET OVERVIEW")
    print("=" * 50)
    print(f"Dataset Shape: {titanic.shape}")
    print(f"Columns: {list(titanic.columns)}")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(titanic.head())
    
    # Dataset info
    print("\nDataset Info:")
    print(titanic.info())
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(titanic.describe())
    
    # Missing values
    print("\nMissing Values:")
    print(titanic.isnull().sum())
    
    return titanic

def create_survival_visualizations(df):
    """Create visualizations focused on survival analysis"""
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Titanic Survival Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall Survival Rate
    survival_counts = df['survived'].value_counts()
    survival_labels = ['Died', 'Survived']
    colors = ['#ff6b6b', '#4ecdc4']
    
    axes[0, 0].pie(survival_counts.values, labels=survival_labels, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Overall Survival Rate')
    
    # 2. Survival by Gender
    survival_gender = pd.crosstab(df['sex'], df['survived'], normalize='index') * 100
    survival_gender.plot(kind='bar', ax=axes[0, 1], color=['#ff6b6b', '#4ecdc4'])
    axes[0, 1].set_title('Survival Rate by Gender')
    axes[0, 1].set_xlabel('Gender')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].legend(['Died', 'Survived'])
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Survival by Passenger Class
    survival_class = pd.crosstab(df['class'], df['survived'], normalize='index') * 100
    survival_class.plot(kind='bar', ax=axes[0, 2], color=['#ff6b6b', '#4ecdc4'])
    axes[0, 2].set_title('Survival Rate by Passenger Class')
    axes[0, 2].set_xlabel('Passenger Class')
    axes[0, 2].set_ylabel('Percentage')
    axes[0, 2].legend(['Died', 'Survived'])
    axes[0, 2].tick_params(axis='x', rotation=0)
    
    # 4. Age Distribution by Survival
    axes[1, 0].hist([df[df['survived']==0]['age'].dropna(), 
                     df[df['survived']==1]['age'].dropna()], 
                    bins=20, label=['Died', 'Survived'], 
                    color=['#ff6b6b', '#4ecdc4'], alpha=0.7)
    axes[1, 0].set_title('Age Distribution by Survival Status')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 5. Survival by Embarkation Port
    embark_survival = pd.crosstab(df['embark_town'], df['survived'], normalize='index') * 100
    embark_survival.plot(kind='bar', ax=axes[1, 1], color=['#ff6b6b', '#4ecdc4'])
    axes[1, 1].set_title('Survival Rate by Embarkation Port')
    axes[1, 1].set_xlabel('Embarkation Port')
    axes[1, 1].set_ylabel('Percentage')
    axes[1, 1].legend(['Died', 'Survived'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Fare Distribution by Survival
    axes[1, 2].boxplot([df[df['survived']==0]['fare'].dropna(), 
                        df[df['survived']==1]['fare'].dropna()], 
                       labels=['Died', 'Survived'])
    axes[1, 2].set_title('Fare Distribution by Survival Status')
    axes[1, 2].set_xlabel('Survival Status')
    axes[1, 2].set_ylabel('Fare')
    
    plt.tight_layout()
    plt.show()

def create_demographic_analysis(df):
    """Create visualizations for demographic analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Demographic Analysis of Titanic Passengers', fontsize=16, fontweight='bold')
    
    # 1. Age vs Fare colored by Survival
    scatter = axes[0, 0].scatter(df['age'], df['fare'], 
                                c=df['survived'], alpha=0.6, 
                                cmap='RdYlBu', s=50)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Fare')
    axes[0, 0].set_title('Age vs Fare (colored by Survival)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Survived')
    
    # 2. Family Size Analysis
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    family_survival = df.groupby('family_size')['survived'].agg(['count', 'sum'])
    family_survival['survival_rate'] = family_survival['sum'] / family_survival['count'] * 100
    
    axes[0, 1].bar(family_survival.index, family_survival['survival_rate'], 
                   color='skyblue', alpha=0.7)
    axes[0, 1].set_xlabel('Family Size')
    axes[0, 1].set_ylabel('Survival Rate (%)')
    axes[0, 1].set_title('Survival Rate by Family Size')
    
    # 3. Heatmap of Survival by Class and Gender
    survival_heatmap = pd.crosstab([df['class'], df['sex']], df['survived'])
    sns.heatmap(survival_heatmap, annot=True, fmt='d', cmap='RdYlBu', 
                ax=axes[1, 0], cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('Survival Count by Class and Gender')
    
    # 4. Age Group Analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    age_group_survival = pd.crosstab(df['age_group'], df['survived'], normalize='index') * 100
    age_group_survival.plot(kind='bar', ax=axes[1, 1], color=['#ff6b6b', '#4ecdc4'])
    axes[1, 1].set_title('Survival Rate by Age Group')
    axes[1, 1].set_xlabel('Age Group')
    axes[1, 1].set_ylabel('Survival Rate (%)')
    axes[1, 1].legend(['Died', 'Survived'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_correlation_analysis(df):
    """Create correlation analysis and advanced visualizations"""
    
    # Prepare numerical data for correlation
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Create correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = numerical_df.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Pairplot for key variables
    key_vars = ['survived', 'age', 'fare', 'sibsp', 'parch']
    if all(var in df.columns for var in key_vars):
        plt.figure(figsize=(12, 10))
        sns.pairplot(df[key_vars], hue='survived', diag_kind='hist', 
                     palette=['#ff6b6b', '#4ecdc4'])
        plt.suptitle('Pairplot of Key Variables', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

def generate_insights(df):
    """Generate and display key insights from the analysis"""
    
    print("=" * 60)
    print("KEY INSIGHTS FROM TITANIC DATASET ANALYSIS")
    print("=" * 60)
    
    # Overall survival rate
    survival_rate = df['survived'].mean() * 100
    print(f"📊 Overall Survival Rate: {survival_rate:.1f}%")
    
    # Gender-based survival
    gender_survival = df.groupby('sex')['survived'].mean() * 100
    print(f"\n👥 Survival by Gender:")
    for gender, rate in gender_survival.items():
        print(f"   {gender.title()}: {rate:.1f}%")
    
    # Class-based survival
    class_survival = df.groupby('class')['survived'].mean() * 100
    print(f"\n🎫 Survival by Passenger Class:")
    for pclass, rate in class_survival.items():
        print(f"   {pclass}: {rate:.1f}%")
    
    # Age analysis
    survived_age = df[df['survived']==1]['age'].mean()
    died_age = df[df['survived']==0]['age'].mean()
    print(f"\n🎂 Average Age:")
    print(f"   Survivors: {survived_age:.1f} years")
    print(f"   Non-survivors: {died_age:.1f} years")
    
    # Family size impact
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    family_survival = df.groupby('family_size')['survived'].mean() * 100
    best_family_size = family_survival.idxmax()
    print(f"\n👨‍👩‍👧‍👦 Family Size with Highest Survival Rate: {best_family_size} members ({family_survival.max():.1f}%)")
    
    # Fare analysis
    survived_fare = df[df['survived']==1]['fare'].mean()
    died_fare = df[df['survived']==0]['fare'].mean()
    print(f"\n💰 Average Fare Paid:")
    print(f"   Survivors: ${survived_fare:.2f}")
    print(f"   Non-survivors: ${died_fare:.2f}")
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS:")
    print("• Women had significantly higher survival rates than men")
    print("• Higher class passengers had better survival chances")
    print("• Children and young adults had better survival rates")
    print("• Passengers who paid higher fares were more likely to survive")
    print("• Small to medium family sizes had optimal survival rates")
    print("=" * 60)

def main():
    """Main function to run the complete analysis"""
    print("🚢 TITANIC DATASET ANALYSIS AND VISUALIZATION")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    print("\n🎨 Creating visualizations...")
    
    # Create different types of visualizations
    create_survival_visualizations(df)
    create_demographic_analysis(df)
    create_correlation_analysis(df)
    
    # Generate insights
    generate_insights(df)
    
    print("\n✅ Analysis completed successfully!")
    print("📈 All visualizations have been generated and key insights extracted.")

# Run the analysis
if __name__ == "__main__":
    main()

# Additional utility functions for custom analysis

def custom_analysis(df, variable):
    """Perform custom analysis on any variable"""
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in dataset")
        return
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Distribution
    plt.subplot(1, 2, 1)
    if df[variable].dtype in ['object', 'category']:
        df[variable].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f'Distribution of {variable}')
    else:
        plt.hist(df[variable].dropna(), bins=20, color='skyblue', alpha=0.7)
        plt.title(f'Distribution of {variable}')
    
    # Subplot 2: Relationship with survival
    plt.subplot(1, 2, 2)
    if df[variable].dtype in ['object', 'category']:
        survival_by_var = pd.crosstab(df[variable], df['survived'], normalize='index') * 100
        survival_by_var.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
        plt.title(f'Survival Rate by {variable}')
    else:
        plt.boxplot([df[df['survived']==0][variable].dropna(), 
                     df[df['survived']==1][variable].dropna()],
                   labels=['Died', 'Survived'])
        plt.title(f'{variable} by Survival Status')
    
    plt.tight_layout()
    plt.show()

# Example usage of custom analysis:
# df = sns.load_dataset('titanic')
# custom_analysis(df, 'deck')  # Analyze deck variable
# custom_analysis(df, 'alone')  # Analyze alone variable