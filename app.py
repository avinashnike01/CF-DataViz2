"""
Streamlit Modern Dashboard for Data Visualization
Author: Professional Data Science Team
Description: Interactive multi-language dashboard for CF Registry Data Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="CF Registry Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Debug: Print configuration loaded
print("=" * 80)
print("Dashboard Configuration Loaded Successfully")
print(f"Timestamp: {datetime.now()}")
print("=" * 80)

# ============================================================================
# MULTI-LANGUAGE SUPPORT
# ============================================================================

TRANSLATIONS = {
    'English': {
        'title': 'CF Registry Analytics Dashboard',
        'subtitle': 'Interactive Data Visualization Platform',
        'upload_file': 'Upload Data File (CSV or Excel)',
        'language': 'Select Language',
        'overview': 'Overview',
        'clinical_outcomes': 'Clinical Outcomes',
        'center_comparison': 'Center Comparison',
        'utilization': 'Utilization',
        
        # KPIs
        'total_patients': 'Total Patients in Registry',
        'total_centers': 'Total Centers',
        'avg_clinic_visits': 'Avg. Clinic Visits per Patient',
        'mortality_rate': 'Mortality Rate (%)',
        'modulator_coverage': 'Modulator Therapy Coverage (%)',
        
        # Overview Tab
        'patients_by_center': 'Patients by Center',
        'patients_by_country': 'Patients by Country',
        'patients_trend': 'Patient Registry Trend Over Years',
        'visits_vs_hospitalizations': 'Clinic Visits vs Hospitalizations',
        
        # Clinical Outcomes Tab
        'fev1_comparison': 'FEV1% Comparison (10 vs 18 year olds)',
        'bmi_distribution': 'BMI Distribution by Age Group',
        'clinical_metrics_center': 'Clinical Metrics by Center',
        'weight_length_percentile': 'Weight-for-Length Percentile Distribution',
        
        # Center Comparison Tab
        'center_performance': 'Center Performance Overview',
        'outcomes_comparison': 'Clinical Outcomes Comparison',
        'utilization_comparison': 'Utilization Metrics Comparison',
        
        # Utilization Tab
        'visits_by_center': 'Clinic Visits by Center',
        'hospitalization_events': 'Hospitalization Events',
        'visits_per_patient': 'Visits per Patient Ratio',
        'utilization_heatmap': 'Utilization Heatmap',
        
        # Labels
        'center': 'Center',
        'country': 'Country',
        'year': 'Year',
        'patients': 'Patients',
        'visits': 'Clinic Visits',
        'hospitalizations': 'Hospitalizations',
        'deceased': 'Deceased',
        'select_all': 'Select All Centers',
        'select_metric': 'Select Metric',
        'no_data': 'No data available. Please upload a file.',
    },
    'Spanish': {
        'title': 'Panel de Análisis del Registro de FQ',
        'subtitle': 'Plataforma Interactiva de Visualización de Datos',
        'upload_file': 'Cargar Archivo de Datos (CSV o Excel)',
        'language': 'Seleccionar Idioma',
        'overview': 'Resumen',
        'clinical_outcomes': 'Resultados Clínicos',
        'center_comparison': 'Comparación de Centros',
        'utilization': 'Utilización',
        
        # KPIs
        'total_patients': 'Total de Pacientes en Registro',
        'total_centers': 'Total de Centros',
        'avg_clinic_visits': 'Promedio de Visitas por Paciente',
        'mortality_rate': 'Tasa de Mortalidad (%)',
        'modulator_coverage': 'Cobertura de Terapia Moduladora (%)',
        
        # Overview Tab
        'patients_by_center': 'Pacientes por Centro',
        'patients_by_country': 'Pacientes por País',
        'patients_trend': 'Tendencia de Registro de Pacientes',
        'visits_vs_hospitalizations': 'Visitas vs Hospitalizaciones',
        
        # Clinical Outcomes Tab
        'fev1_comparison': 'Comparación FEV1% (10 vs 18 años)',
        'bmi_distribution': 'Distribución de IMC por Grupo de Edad',
        'clinical_metrics_center': 'Métricas Clínicas por Centro',
        'weight_length_percentile': 'Distribución de Percentil Peso-Longitud',
        
        # Center Comparison Tab
        'center_performance': 'Rendimiento General de Centros',
        'outcomes_comparison': 'Comparación de Resultados Clínicos',
        'utilization_comparison': 'Comparación de Métricas de Utilización',
        
        # Utilization Tab
        'visits_by_center': 'Visitas Clínicas por Centro',
        'hospitalization_events': 'Eventos de Hospitalización',
        'visits_per_patient': 'Relación Visitas por Paciente',
        'utilization_heatmap': 'Mapa de Calor de Utilización',
        
        # Labels
        'center': 'Centro',
        'country': 'País',
        'year': 'Año',
        'patients': 'Pacientes',
        'visits': 'Visitas Clínicas',
        'hospitalizations': 'Hospitalizaciones',
        'deceased': 'Fallecidos',
        'select_all': 'Seleccionar Todos los Centros',
        'select_metric': 'Seleccionar Métrica',
        'no_data': 'No hay datos disponibles. Por favor, cargue un archivo.',
    },
    'Portuguese': {
        'title': 'Painel de Análise do Registro de FC',
        'subtitle': 'Plataforma Interativa de Visualização de Dados',
        'upload_file': 'Carregar Arquivo de Dados (CSV ou Excel)',
        'language': 'Selecionar Idioma',
        'overview': 'Visão Geral',
        'clinical_outcomes': 'Resultados Clínicos',
        'center_comparison': 'Comparação de Centros',
        'utilization': 'Utilização',
        
        # KPIs
        'total_patients': 'Total de Pacientes no Registro',
        'total_centers': 'Total de Centros',
        'avg_clinic_visits': 'Média de Visitas por Paciente',
        'mortality_rate': 'Taxa de Mortalidade (%)',
        'modulator_coverage': 'Cobertura de Terapia Moduladora (%)',
        
        # Overview Tab
        'patients_by_center': 'Pacientes por Centro',
        'patients_by_country': 'Pacientes por País',
        'patients_trend': 'Tendência de Registro de Pacientes',
        'visits_vs_hospitalizations': 'Visitas vs Hospitalizações',
        
        # Clinical Outcomes Tab
        'fev1_comparison': 'Comparação FEV1% (10 vs 18 anos)',
        'bmi_distribution': 'Distribuição de IMC por Faixa Etária',
        'clinical_metrics_center': 'Métricas Clínicas por Centro',
        'weight_length_percentile': 'Distribuição de Percentil Peso-Comprimento',
        
        # Center Comparison Tab
        'center_performance': 'Desempenho Geral dos Centros',
        'outcomes_comparison': 'Comparação de Resultados Clínicos',
        'utilization_comparison': 'Comparação de Métricas de Utilização',
        
        # Utilization Tab
        'visits_by_center': 'Visitas Clínicas por Centro',
        'hospitalization_events': 'Eventos de Hospitalização',
        'visits_per_patient': 'Proporção Visitas por Paciente',
        'utilization_heatmap': 'Mapa de Calor de Utilização',
        
        # Labels
        'center': 'Centro',
        'country': 'País',
        'year': 'Ano',
        'patients': 'Pacientes',
        'visits': 'Visitas Clínicas',
        'hospitalizations': 'Hospitalizações',
        'deceased': 'Falecidos',
        'select_all': 'Selecionar Todos os Centros',
        'select_metric': 'Selecionar Métrica',
        'no_data': 'Nenhum dado disponível. Por favor, carregue um arquivo.',
    }
}

# Debug: Print available languages
print("Available Languages:", list(TRANSLATIONS.keys()))

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """
    Load data from uploaded CSV or Excel file with caching for performance
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
    
    Returns:
    --------
    pd.DataFrame : Loaded dataframe
    """
    print("\n" + "="*80)
    print("LOADING DATA FROM UPLOADED FILE")
    print("="*80)
    
    try:
        if uploaded_file.name.endswith('.csv'):
            print(f"Reading CSV file: {uploaded_file.name}")
            df = pd.read_csv(uploaded_file)
            print(f"CSV loaded successfully. Shape: {df.shape}")
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            print(f"Reading Excel file: {uploaded_file.name}")
            df = pd.read_excel(uploaded_file)
            print(f"Excel loaded successfully. Shape: {df.shape}")
        else:
            print(f"ERROR: Unsupported file format: {uploaded_file.name}")
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Debug: Print column names and types
        print("\nColumn Names:")
        print(list(df.columns))
        print("\nData Types:")
        print(df.dtypes)
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("="*80 + "\n")
        
        return df
    
    except Exception as e:
        print(f"ERROR loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data
def load_default_data():
    """
    Load default sample data for demonstration
    
    Returns:
    --------
    pd.DataFrame : Default sample dataframe
    """
    print("\n" + "="*80)
    print("LOADING DEFAULT SAMPLE DATA")
    print("="*80)
    
    default_file_path = '/mnt/project/DataViz_test_2.xlsx'
    
    try:
        df = pd.read_excel(default_file_path)
        print(f"Default data loaded. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("="*80 + "\n")
        return df
    except Exception as e:
        print(f"WARNING: Could not load default data: {str(e)}")
        print("="*80 + "\n")
        return None

# ============================================================================
# KPI CALCULATION FUNCTIONS
# ============================================================================

def calculate_kpis(df):
    """
    Calculate Key Performance Indicators from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict : Dictionary containing KPI values
    """
    print("\n" + "="*80)
    print("CALCULATING KPIs")
    print("="*80)
    
    kpis = {}
    
    # Total patients in registry
    kpis['total_patients'] = df['People in the Registry'].sum()
    print(f"Total Patients: {kpis['total_patients']}")
    
    # Total centers
    kpis['total_centers'] = df['Centers'].nunique()
    print(f"Total Centers: {kpis['total_centers']}")
    
    # Average clinic visits per patient
    total_visits = df['Total number of clinic visits'].sum()
    total_patients = df['People in the Registry'].sum()
    kpis['avg_clinic_visits'] = round(total_visits / total_patients, 2) if total_patients > 0 else 0
    print(f"Avg Clinic Visits per Patient: {kpis['avg_clinic_visits']}")
    
    # Mortality rate
    total_deceased = df['Number of People with CF deceased'].sum()
    kpis['mortality_rate'] = round((total_deceased / total_patients) * 100, 2) if total_patients > 0 else 0
    print(f"Mortality Rate: {kpis['mortality_rate']}%")
    
    # Modulator therapy coverage
    total_on_modulator = df['Total number of patients on modulator med'].sum()
    total_qualifying = df['Total number of patients qualifying for modulator therapy'].sum()
    kpis['modulator_coverage'] = round((total_on_modulator / total_qualifying) * 100, 2) if total_qualifying > 0 else 0
    print(f"Modulator Coverage: {kpis['modulator_coverage']}%")
    
    print("="*80 + "\n")
    
    return kpis

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_patients_by_center_chart(df, lang_dict):
    """Create bar chart showing patients by center"""
    print("Creating chart: Patients by Center")
    
    # Aggregate data by center
    center_data = df.groupby('Centers')['People in the Registry'].sum().reset_index()
    center_data = center_data.sort_values('People in the Registry', ascending=True)
    
    print(f"Number of centers in chart: {len(center_data)}")
    
    fig = px.bar(
        center_data,
        x='People in the Registry',
        y='Centers',
        orientation='h',
        title=lang_dict['patients_by_center'],
        labels={'People in the Registry': lang_dict['patients'], 'Centers': lang_dict['center']},
        color='People in the Registry',
        color_continuous_scale='Blues',
        template='plotly_white'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig

def create_patients_by_country_chart(df, lang_dict):
    """Create pie chart showing patients by country"""
    print("Creating chart: Patients by Country")
    
    # Aggregate data by country
    country_data = df.groupby('Country')['People in the Registry'].sum().reset_index()
    
    print(f"Number of countries: {len(country_data)}")
    
    fig = px.pie(
        country_data,
        values='People in the Registry',
        names='Country',
        title=lang_dict['patients_by_country'],
        template='plotly_white',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=500,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50')
    )
    
    return fig

def create_patients_trend_chart(df, lang_dict):
    """Create line chart showing patient registry trend over years"""
    print("Creating chart: Patient Trend Over Years")
    
    # Aggregate data by year
    year_data = df.groupby('Year')['People in the Registry'].sum().reset_index()
    year_data = year_data.sort_values('Year')
    
    print(f"Years covered: {year_data['Year'].min()} to {year_data['Year'].max()}")
    
    fig = px.line(
        year_data,
        x='Year',
        y='People in the Registry',
        title=lang_dict['patients_trend'],
        labels={'People in the Registry': lang_dict['patients'], 'Year': lang_dict['year']},
        markers=True,
        template='plotly_white'
    )
    
    fig.update_traces(line_color='#3498db', marker=dict(size=10, color='#e74c3c'))
    fig.update_layout(
        height=400,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig

def create_visits_vs_hospitalizations_chart(df, lang_dict):
    """Create scatter plot comparing clinic visits and hospitalizations"""
    print("Creating chart: Visits vs Hospitalizations")
    
    fig = px.scatter(
        df,
        x='Total number of clinic visits',
        y='Total number of hospitalization events',
        size='People in the Registry',
        color='Country',
        hover_data=['Centers', 'Year'],
        title=lang_dict['visits_vs_hospitalizations'],
        labels={
            'Total number of clinic visits': lang_dict['visits'],
            'Total number of hospitalization events': lang_dict['hospitalizations']
        },
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    fig.update_layout(
        height=500,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig

def create_fev1_comparison_chart(df, lang_dict):
    """Create grouped bar chart comparing FEV1% for different age groups"""
    print("Creating chart: FEV1% Comparison")
    
    # Aggregate data
    fev1_data = df.groupby('Centers').agg({
        'Median FEV1% for 10 year olds': 'mean',
        'Median FEV1% for 18 year olds': 'mean'
    }).reset_index()
    
    fev1_data = fev1_data.sort_values('Median FEV1% for 10 year olds', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='10 Year Olds',
        x=fev1_data['Median FEV1% for 10 year olds'],
        y=fev1_data['Centers'],
        orientation='h',
        marker=dict(color='#3498db')
    ))
    
    fig.add_trace(go.Bar(
        name='18 Year Olds',
        x=fev1_data['Median FEV1% for 18 year olds'],
        y=fev1_data['Centers'],
        orientation='h',
        marker=dict(color='#e74c3c')
    ))
    
    fig.update_layout(
        title=lang_dict['fev1_comparison'],
        barmode='group',
        height=500,
        template='plotly_white',
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title='FEV1%',
        yaxis_title=lang_dict['center'],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def create_bmi_distribution_chart(df, lang_dict):
    """Create box plot showing BMI distribution by age group"""
    print("Creating chart: BMI Distribution")
    
    # Prepare data for box plot
    bmi_data = []
    
    # BMI for 2-19 years
    for _, row in df.iterrows():
        bmi_data.append({
            'Age Group': '2-19 years',
            'BMI Percentile': row['Median BMI%ile 2-19 years'],
            'Center': row['Centers']
        })
    
    # BMI for 20+ years
    for _, row in df.iterrows():
        bmi_data.append({
            'Age Group': '20+ years',
            'BMI': row['Median BMI for patients 20 years and older'],
            'Center': row['Centers']
        })
    
    bmi_df = pd.DataFrame(bmi_data)
    
    # Create subplots for different age groups
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('BMI Percentile (2-19 years)', 'BMI (20+ years)')
    )
    
    # Add box plot for 2-19 years
    data_2_19 = bmi_df[bmi_df['Age Group'] == '2-19 years']
    fig.add_trace(
        go.Box(y=data_2_19['BMI Percentile'], name='2-19 years', marker_color='#3498db'),
        row=1, col=1
    )
    
    # Add box plot for 20+ years
    data_20_plus = bmi_df[bmi_df['Age Group'] == '20+ years']
    fig.add_trace(
        go.Box(y=data_20_plus['BMI'], name='20+ years', marker_color='#e74c3c'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=lang_dict['bmi_distribution'],
        height=500,
        showlegend=False,
        template='plotly_white',
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50')
    )
    
    return fig

def create_clinical_metrics_heatmap(df, lang_dict):
    """Create heatmap showing clinical metrics by center"""
    print("Creating chart: Clinical Metrics Heatmap")
    
    # Select clinical metrics
    metrics = [
        'Median FEV1% for 10 year olds',
        'Median FEV1% for 18 year olds',
        'Median BMI%ile 2-19 years',
        'Median BMI for patients 20 years and older'
    ]
    
    # Create heatmap data
    heatmap_data = df.groupby('Centers')[metrics].mean()
    
    # Normalize data for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    heatmap_normalized = scaler.fit_transform(heatmap_data)
    
    print(f"Heatmap shape: {heatmap_normalized.shape}")
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_normalized.T,
        x=heatmap_data.index,
        y=[m.replace('Median ', '').replace(' for patients', '') for m in metrics],
        colorscale='RdYlBu_r',
        text=heatmap_data.T.values,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        colorbar=dict(title="Normalized<br>Score")
    ))
    
    fig.update_layout(
        title=lang_dict['clinical_metrics_center'],
        height=400,
        template='plotly_white',
        font=dict(size=11),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title=lang_dict['center'],
        yaxis_title='Metric'
    )
    
    return fig

def create_weight_length_chart(df, lang_dict):
    """Create histogram showing weight-for-length percentile distribution"""
    print("Creating chart: Weight-for-Length Percentile Distribution")
    
    fig = px.histogram(
        df,
        x='Median WHO Weight-for-Length Percentile for patients less than 24 months',
        nbins=20,
        title=lang_dict['weight_length_percentile'],
        labels={'Median WHO Weight-for-Length Percentile for patients less than 24 months': 'Weight-for-Length Percentile'},
        template='plotly_white',
        color_discrete_sequence=['#9b59b6']
    )
    
    fig.update_layout(
        height=400,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title_font=dict(size=14),
        yaxis_title='Frequency'
    )
    
    return fig

def create_center_performance_chart(df, lang_dict):
    """Create radar chart showing center performance across multiple metrics"""
    print("Creating chart: Center Performance Radar")
    
    # Calculate performance metrics per center
    center_metrics = df.groupby('Centers').agg({
        'People in the Registry': 'sum',
        'Total number of clinic visits': 'sum',
        'Median FEV1% for 18 year olds': 'mean',
        'Median BMI%ile 2-19 years': 'mean',
        'Total number of patients on modulator med': 'sum'
    }).reset_index()
    
    # Normalize metrics to 0-100 scale for radar chart
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    
    metrics_cols = [
        'People in the Registry',
        'Total number of clinic visits',
        'Median FEV1% for 18 year olds',
        'Median BMI%ile 2-19 years',
        'Total number of patients on modulator med'
    ]
    
    center_metrics[metrics_cols] = scaler.fit_transform(center_metrics[metrics_cols])
    
    # Create radar chart for top 5 centers by patient count
    top_centers = df.groupby('Centers')['People in the Registry'].sum().nlargest(5).index
    
    fig = go.Figure()
    
    for center in top_centers:
        center_data = center_metrics[center_metrics['Centers'] == center].iloc[0]
        
        fig.add_trace(go.Scatterpolar(
            r=[
                center_data['People in the Registry'],
                center_data['Total number of clinic visits'],
                center_data['Median FEV1% for 18 year olds'],
                center_data['Median BMI%ile 2-19 years'],
                center_data['Total number of patients on modulator med']
            ],
            theta=['Patients', 'Visits', 'FEV1%', 'BMI%', 'Modulator Therapy'],
            fill='toself',
            name=center
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title=lang_dict['center_performance'],
        height=500,
        template='plotly_white',
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50')
    )
    
    return fig

def create_utilization_heatmap(df, lang_dict):
    """Create heatmap showing utilization metrics"""
    print("Creating chart: Utilization Heatmap")
    
    # Calculate visits per patient and hospitalization rate
    df_util = df.copy()
    df_util['Visits per Patient'] = df_util['Total number of clinic visits'] / df_util['People in the Registry']
    df_util['Hospitalization Rate'] = df_util['Total number of hospitalization events'] / df_util['People in the Registry']
    
    # Group by center and year
    util_pivot = df_util.pivot_table(
        values='Visits per Patient',
        index='Centers',
        columns='Year',
        aggfunc='mean'
    )
    
    print(f"Utilization heatmap shape: {util_pivot.shape}")
    
    fig = go.Figure(data=go.Heatmap(
        z=util_pivot.values,
        x=util_pivot.columns,
        y=util_pivot.index,
        colorscale='Viridis',
        text=util_pivot.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Visits per<br>Patient")
    ))
    
    fig.update_layout(
        title=lang_dict['utilization_heatmap'],
        height=500,
        template='plotly_white',
        font=dict(size=11),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title=lang_dict['year'],
        yaxis_title=lang_dict['center']
    )
    
    return fig

def create_visits_by_center_chart(df, lang_dict):
    """Create stacked bar chart showing visits by center and year"""
    print("Creating chart: Visits by Center")
    
    visits_data = df.groupby(['Centers', 'Year'])['Total number of clinic visits'].sum().reset_index()
    
    fig = px.bar(
        visits_data,
        x='Centers',
        y='Total number of clinic visits',
        color='Year',
        title=lang_dict['visits_by_center'],
        labels={'Total number of clinic visits': lang_dict['visits'], 'Centers': lang_dict['center']},
        template='plotly_white',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=500,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        xaxis_tickangle=-45
    )
    
    return fig

def create_hospitalization_events_chart(df, lang_dict):
    """Create line chart showing hospitalization events over time"""
    print("Creating chart: Hospitalization Events")
    
    hosp_data = df.groupby('Year')['Total number of hospitalization events'].sum().reset_index()
    
    fig = px.area(
        hosp_data,
        x='Year',
        y='Total number of hospitalization events',
        title=lang_dict['hospitalization_events'],
        labels={'Total number of hospitalization events': lang_dict['hospitalizations'], 'Year': lang_dict['year']},
        template='plotly_white',
        color_discrete_sequence=['#e74c3c']
    )
    
    fig.update_layout(
        height=400,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig

def create_visits_per_patient_chart(df, lang_dict):
    """Create bar chart showing visits per patient ratio by center"""
    print("Creating chart: Visits per Patient")
    
    df_ratio = df.copy()
    df_ratio['Visits per Patient'] = df_ratio['Total number of clinic visits'] / df_ratio['People in the Registry']
    
    ratio_data = df_ratio.groupby('Centers')['Visits per Patient'].mean().reset_index()
    ratio_data = ratio_data.sort_values('Visits per Patient', ascending=True)
    
    fig = px.bar(
        ratio_data,
        x='Visits per Patient',
        y='Centers',
        orientation='h',
        title=lang_dict['visits_per_patient'],
        labels={'Visits per Patient': 'Visits per Patient', 'Centers': lang_dict['center']},
        template='plotly_white',
        color='Visits per Patient',
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        font=dict(size=12),
        title_font=dict(size=16, color='#2c3e50'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS for modern dashboard styling"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 3px solid #3498db;
    }
    
    h2 {
        color: #34495e;
        font-weight: 600;
    }
    
    h3 {
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* KPI card styling */
    .kpi-card {
        # background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .kpi-value {
        font-size: 36px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .kpi-label {
        font-size: 14px;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: transparent;
        border-radius: 8px;
        font-weight: 600;
        color: #7f8c8d;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        background-color: #ecf0f1;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    /* Metric card enhancement */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #2c3e50;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    print("\n" + "="*80)
    print("STARTING STREAMLIT DASHBOARD APPLICATION")
    print(f"Timestamp: {datetime.now()}")
    print("="*80 + "\n")
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
        print(f"Initialized language: {st.session_state.language}")
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/combo-chart.png", width=80)
        st.title("⚙️ Configuration")
        
        # Language selection
        selected_language = st.selectbox(
            "🌐 " + TRANSLATIONS[st.session_state.language]['language'],
            list(TRANSLATIONS.keys()),
            index=list(TRANSLATIONS.keys()).index(st.session_state.language)
        )
        
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            print(f"Language changed to: {selected_language}")
            st.rerun()
        
        st.markdown("---")
        
        # File uploader
        st.subheader("📁 " + TRANSLATIONS[st.session_state.language]['upload_file'])
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your data file to visualize"
        )
        
        st.markdown("---")
        st.info("💡 **Tip:** Upload a file with similar structure to the sample data for best results.")
        
        # About section
        with st.expander("ℹ️ About this Dashboard"):
            st.write("""
            This interactive dashboard provides comprehensive analytics for CF Registry data including:
            - Patient demographics and trends
            - Clinical outcomes monitoring
            - Center performance comparison
            - Utilization metrics analysis
            
            **Features:**
            - Multi-language support (EN, ES, PT)
            - Interactive visualizations
            - Real-time data updates
            - Export capabilities
            """)
    
    # Get language dictionary
    lang_dict = TRANSLATIONS[st.session_state.language]
    
    # Main header
    st.title("📊 " + lang_dict['title'])
    st.markdown(f"### {lang_dict['subtitle']}")
    st.markdown("---")
    
    # Load data
    if uploaded_file is not None:
        print(f"\nUser uploaded file: {uploaded_file.name}")
        df = load_data(uploaded_file)
    else:
        print("\nNo file uploaded, loading default data")
        df = load_default_data()
    
    # Check if data is loaded
    if df is None or df.empty:
        st.warning(lang_dict['no_data'])
        st.info("👆 Please upload a data file using the sidebar")
        return
    
    print(f"\nData loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Display KPIs
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label=lang_dict['total_patients'],
            value=f"{kpis['total_patients']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label=lang_dict['total_centers'],
            value=f"{kpis['total_centers']}",
            delta=None
        )
    
    with col3:
        st.metric(
            label=lang_dict['avg_clinic_visits'],
            value=f"{kpis['avg_clinic_visits']:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label=lang_dict['mortality_rate'],
            value=f"{kpis['mortality_rate']}%",
            delta=None,
            delta_color="inverse"
        )
    
    with col5:
        st.metric(
            label=lang_dict['modulator_coverage'],
            value=f"{kpis['modulator_coverage']}%",
            delta=None
        )
    
    st.markdown("---")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 " + lang_dict['overview'],
        "🏥 " + lang_dict['clinical_outcomes'],
        "🔄 " + lang_dict['center_comparison'],
        "📈 " + lang_dict['utilization']
    ])
    
    # ========================================================================
    # OVERVIEW TAB
    # ========================================================================
    with tab1:
        print("\n" + "-"*80)
        print("RENDERING OVERVIEW TAB")
        print("-"*80)
        
        st.markdown("### " + lang_dict['overview'])
        
        # First row - Patients by Center and Country
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_patients_by_center_chart(df, lang_dict)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_patients_by_country_chart(df, lang_dict)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Second row - Patient Trend
        fig3 = create_patients_trend_chart(df, lang_dict)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Third row - Visits vs Hospitalizations
        fig4 = create_visits_vs_hospitalizations_chart(df, lang_dict)
        st.plotly_chart(fig4, use_container_width=True)
    
    # ========================================================================
    # CLINICAL OUTCOMES TAB
    # ========================================================================
    with tab2:
        print("\n" + "-"*80)
        print("RENDERING CLINICAL OUTCOMES TAB")
        print("-"*80)
        
        st.markdown("### " + lang_dict['clinical_outcomes'])
        
        # First row - FEV1 Comparison
        fig5 = create_fev1_comparison_chart(df, lang_dict)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Second row - BMI Distribution and Weight-Length
        col1, col2 = st.columns(2)
        
        with col1:
            fig6 = create_bmi_distribution_chart(df, lang_dict)
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            fig7 = create_weight_length_chart(df, lang_dict)
            st.plotly_chart(fig7, use_container_width=True)
        
        # Third row - Clinical Metrics Heatmap
        fig8 = create_clinical_metrics_heatmap(df, lang_dict)
        st.plotly_chart(fig8, use_container_width=True)
    
    # ========================================================================
    # CENTER COMPARISON TAB
    # ========================================================================
    with tab3:
        print("\n" + "-"*80)
        print("RENDERING CENTER COMPARISON TAB")
        print("-"*80)
        
        st.markdown("### " + lang_dict['center_comparison'])
        
        # # Center Performance Radar Chart
        # fig9 = create_center_performance_chart(df, lang_dict)
        # st.plotly_chart(fig9, use_container_width=True)
        
        # Interactive center selection for detailed comparison
        st.markdown("#### Detailed Center Analysis")
        
        selected_centers = st.multiselect(
            "Select centers to compare:",
            options=df['Centers'].unique().tolist(),
            default=df['Centers'].unique().tolist()[:3]
        )
        
        if selected_centers:
            filtered_df = df[df['Centers'].isin(selected_centers)]
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig10 = create_patients_by_center_chart(filtered_df, lang_dict)
                st.plotly_chart(fig10, use_container_width=True)
            
            with col2:
                fig11 = create_fev1_comparison_chart(filtered_df, lang_dict)
                st.plotly_chart(fig11, use_container_width=True)
    
    # ========================================================================
    # UTILIZATION TAB
    # ========================================================================
    with tab4:
        print("\n" + "-"*80)
        print("RENDERING UTILIZATION TAB")
        print("-"*80)
        
        st.markdown("### " + lang_dict['utilization'])
        
        # First row - Visits by Center
        fig12 = create_visits_by_center_chart(df, lang_dict)
        st.plotly_chart(fig12, use_container_width=True)
        
        # Second row - Hospitalizations and Visits per Patient
        col1, col2 = st.columns(2)
        
        with col1:
            fig13 = create_hospitalization_events_chart(df, lang_dict)
            st.plotly_chart(fig13, use_container_width=True)
        
        with col2:
            fig14 = create_visits_per_patient_chart(df, lang_dict)
            st.plotly_chart(fig14, use_container_width=True)
        
        # Third row - Utilization Heatmap
        fig15 = create_utilization_heatmap(df, lang_dict)
        st.plotly_chart(fig15, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>CF Registry Analytics Dashboard | Powered by Streamlit & Plotly</p>
        <p>© 2024 Professional Data Science Team | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)
    
    print("\n" + "="*80)
    print("DASHBOARD RENDERING COMPLETE")
    print("="*80 + "\n")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
