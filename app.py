#!/usr/bin/env python3
"""
PatSnap Pharma Intelligence — Hugging Face Space Demo
Product-grade multi-module AI agent for life science intelligence.

Modules:
  - Agent Chat (natural language → tool orchestration → report)
  - Target Intelligence    (靶点全景)
  - Drug Exploration       (药物管线)
  - Disease Investigation  (疾病格局)
  - Company Profiling      (公司分析)
  - Clinical Trials        (临床试验)
"""

import os, json, asyncio, time, re
from collections import Counter
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import gradio as gr

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = os.getenv("PATSNAP_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # optional: enable LLM agent mode
SERVER_URL = f"https://connect.patsnap.com/096456/logic-mcp?apiKey={API_KEY}"

# Module definitions
MODULES = {
    "target":  {"icon": "🎯", "label": "Target Intelligence", "label_cn": "靶点全景",
                "tool": "ls_target_search", "entity": "target",
                "desc": "Analyze any biomedical target — biology, drugs, pipeline, trials."},
    "drug":    {"icon": "💊", "label": "Drug Exploration", "label_cn": "药物管线",
                "tool": "ls_drug_search", "entity": "drug",
                "desc": "Search drugs by target, disease, mechanism, or company."},
    "disease": {"icon": "🏥", "label": "Disease Investigation", "label_cn": "疾病格局",
                "tool": "ls_disease_search", "entity": "disease",
                "desc": "Understand disease landscape — epidemiology, treatments, pipeline."},
    "company": {"icon": "🏢", "label": "Company Profiling", "label_cn": "公司分析",
                "tool": "ls_company_search", "entity": "company",
                "desc": "Profile pharma companies — pipeline, deals, therapeutic focus."},
    "trial":   {"icon": "🧪", "label": "Clinical Trials", "label_cn": "临床试验",
                "tool": "ls_clinical_trial_search", "entity": "trial",
                "desc": "Explore clinical trials by target, disease, phase, or sponsor."},
}

# Language: 'en' or 'zh'
DEFAULT_LANG = "en"

# =============================================================================
# MOCK DATA — Comprehensive, module-specific
# =============================================================================

MOCK_TARGETS = {
    "EGFR": {
        "name": "EGFR", "full_name": "Epidermal Growth Factor Receptor",
        "family": "ErbB/HER receptor tyrosine kinase",
        "class": "Kinase", "indication_count": 12, "drug_count": 28,
        "pdb_ids": ["1M17", "1XKK", "2J5F"],
        "pathways": ["RAS/RAF/MEK/ERK", "PI3K/AKT/mTOR", "JAK/STAT"],
        "mutation_hotspots": ["L858R (exon 21)", "exon 19 deletion", "T790M (exon 20)", "C797S (exon 20)"],
        "approved_drugs": [
            {"name": "Osimertinib", "type": "Small molecule", "gen": "3rd-gen TKI",
             "year": 2015, "indications": ["NSCLC (T790M+)", "NSCLC (1L EGFRm)", "NSCLC (adjuvant)"],
             "company": "AstraZeneca"},
            {"name": "Gefitinib", "type": "Small molecule", "gen": "1st-gen TKI",
             "year": 2003, "indications": ["NSCLC (EGFRm)"], "company": "AstraZeneca"},
            {"name": "Erlotinib", "type": "Small molecule", "gen": "1st-gen TKI",
             "year": 2004, "indications": ["NSCLC", "Pancreatic"], "company": "Roche/Genentech"},
            {"name": "Afatinib", "type": "Small molecule", "gen": "2nd-gen TKI",
             "year": 2013, "indications": ["NSCLC (EGFRm)"], "company": "Boehringer Ingelheim"},
            {"name": "Dacomitinib", "type": "Small molecule", "gen": "2nd-gen TKI",
             "year": 2018, "indications": ["NSCLC (EGFRm)"], "company": "Pfizer"},
            {"name": "Cetuximab", "type": "Monoclonal antibody", "gen": "mAb",
             "year": 2004, "indications": ["CRC", "HNSCC"], "company": "Merck KGaA / BMS"},
            {"name": "Panitumumab", "type": "Monoclonal antibody", "gen": "mAb",
             "year": 2006, "indications": ["CRC"], "company": "Amgen"},
            {"name": "Amivantamab", "type": "Bispecific antibody", "gen": "BsAb",
             "year": 2021, "indications": ["NSCLC (ex20ins)"], "company": "Janssen"},
            {"name": "Patritumab deruxtecan", "type": "ADC", "gen": "ADC (HER3)",
             "year": 2024, "indications": ["NSCLC (post-TKI)"], "company": "Daiichi Sankyo / Merck"},
        ],
        "pipeline_summary": {
            "phase_3": 45, "phase_2": 120, "phase_1": 85, "preclinical": 200,
            "hot_topics": ["4th-gen TKIs (C797S)", "Bispecific ADCs", "PROTAC degraders",
                          "Combination with immunotherapy", "Brain-penetrant TKIs"],
        },
        "competitive_landscape": "Highly competitive — every major pharma has an EGFR asset. "
                                 "Innovation is focused on resistance mechanisms and next-gen modalities.",
    },
    "HER2": {
        "name": "HER2", "full_name": "Human Epidermal Growth Factor Receptor 2",
        "family": "ErbB/HER receptor tyrosine kinase",
        "class": "Kinase", "indication_count": 5, "drug_count": 15,
        "pdb_ids": ["1N8Z", "3PP0"],
        "pathways": ["RAS/RAF/MEK/ERK", "PI3K/AKT/mTOR"],
        "mutation_hotspots": ["Amplification (breast/gastric)", "Exon 20 mutations (NSCLC)"],
        "approved_drugs": [
            {"name": "Trastuzumab", "type": "Monoclonal antibody", "gen": "mAb",
             "year": 1998, "indications": ["HER2+ Breast Cancer", "HER2+ Gastric Cancer"], "company": "Roche"},
            {"name": "Trastuzumab deruxtecan", "type": "ADC", "gen": "ADC",
             "year": 2019, "indications": ["HER2+ Breast Cancer", "HER2-low Breast Cancer", "HER2+ Gastric Cancer",
                                          "HER2-mutant NSCLC"], "company": "Daiichi Sankyo / AstraZeneca"},
            {"name": "Pertuzumab", "type": "Monoclonal antibody", "gen": "mAb",
             "year": 2012, "indications": ["HER2+ Breast Cancer"], "company": "Roche"},
            {"name": "Lapatinib", "type": "Small molecule", "gen": "TKI",
             "year": 2007, "indications": ["HER2+ Breast Cancer"], "company": "Novartis"},
            {"name": "Tucatinib", "type": "Small molecule", "gen": "TKI",
             "year": 2020, "indications": ["HER2+ Breast Cancer (CNS mets)"], "company": "Seagen / Merck"},
        ],
        "pipeline_summary": {
            "phase_3": 25, "phase_2": 80, "phase_1": 55, "preclinical": 130,
            "hot_topics": ["HER2-low targeting", "Bispecific ADCs", "Brain metastasis"],
        },
        "competitive_landscape": "HER2 ADC space is the current battleground. Enhertu dominates; "
                                 "competitors focus on differentiation via payload, DAR, or epitope.",
    },
    "PD-L1": {
        "name": "PD-L1", "full_name": "Programmed Death-Ligand 1",
        "family": "B7 immune checkpoint",
        "class": "Immune checkpoint ligand", "indication_count": 20, "drug_count": 20,
        "approved_drugs": [
            {"name": "Atezolizumab", "type": "Monoclonal antibody", "gen": "Anti-PD-L1 mAb",
             "year": 2016, "indications": ["NSCLC", "SCLC", "Urothelial", "HCC"], "company": "Roche"},
            {"name": "Durvalumab", "type": "Monoclonal antibody", "gen": "Anti-PD-L1 mAb",
             "year": 2017, "indications": ["NSCLC (Stage III)", "SCLC", "Biliary Tract"], "company": "AstraZeneca"},
            {"name": "Avelumab", "type": "Monoclonal antibody", "gen": "Anti-PD-L1 mAb",
             "year": 2017, "indications": ["Merkel Cell", "Urothelial", "RCC"], "company": "Merck KGaA / Pfizer"},
        ],
        "pipeline_summary": {"phase_3": 35, "phase_2": 60, "phase_1": 40, "preclinical": 100},
        "competitive_landscape": "PD-L1 is a companion to PD-1 — the focus is on combination strategies "
                                 "and predictive biomarker development.",
    },
}

MOCK_COMPANIES = {
    "Roche": {
        "name": "Roche", "ticker": "ROG.SW",
        "headquarters": "Basel, Switzerland",
        "employees": "~100,000",
        "2024_revenue": "$65.4B",
        "therapeutic_areas": ["Oncology", "Neuroscience", "Ophthalmology", "Immunology", "Infectious Disease"],
        "flagship_drugs": [
            {"name": "Trastuzumab (Herceptin)", "target": "HER2", "sales": "$3.2B", "phase": "Approved"},
            {"name": "Atezolizumab (Tecentriq)", "target": "PD-L1", "sales": "$4.8B", "phase": "Approved"},
            {"name": "Bevacizumab (Avastin)", "target": "VEGF", "sales": "$2.1B", "phase": "Approved"},
            {"name": "Trastuzumab deruxtecan (co-developed)", "target": "HER2", "sales": "$3.5B", "phase": "Approved"},
        ],
        "pipeline_count": {"approved": 28, "phase_3": 15, "phase_2": 35, "phase_1": 22},
        "recent_deals": [
            "Acquired Carmot Therapeutics (obesity) — $2.7B upfront (2023)",
            "Acquired Telavant (IBD) — $7.1B (2023)",
        ],
        "strategy": "Roche combines strong internal R&D with strategic bolt-on acquisitions. "
                     "Oncology remains the core, with growing investment in immunology and metabolic disease.",
    },
    "AstraZeneca": {
        "name": "AstraZeneca", "ticker": "AZN.L",
        "headquarters": "Cambridge, UK",
        "employees": "~90,000",
        "2024_revenue": "$54.1B",
        "therapeutic_areas": ["Oncology", "CVRM", "Respiratory & Immunology", "Rare Disease"],
        "flagship_drugs": [
            {"name": "Osimertinib (Tagrisso)", "target": "EGFR", "sales": "$6.8B", "phase": "Approved"},
            {"name": "Durvalumab (Imfinzi)", "target": "PD-L1", "sales": "$4.5B", "phase": "Approved"},
            {"name": "Trastuzumab deruxtecan (co-developed)", "target": "HER2", "sales": "$3.5B", "phase": "Approved"},
            {"name": "Dapagliflozin (Farxiga)", "target": "SGLT2", "sales": "$7.1B", "phase": "Approved"},
        ],
        "pipeline_count": {"approved": 22, "phase_3": 12, "phase_2": 28, "phase_1": 18},
        "recent_deals": [
            "Acquired Gracell Biotechnologies (CAR-T) — $1.2B (2023)",
            "Acquired Icosavax (RSV/hMPV vaccine) — $1.1B (2023)",
            "Acquired Fusion Pharmaceuticals (radiopharma) — $2.4B (2024)",
        ],
        "strategy": "AZ's oncology portfolio is anchored by Tagrisso, Imfinzi, and Enhertu. "
                     "Actively expanding into cell therapy, radiopharmaceuticals, and ADCs.",
    },
}

MOCK_DISEASES = {
    "NSCLC": {
        "name": "Non-Small Cell Lung Cancer",
        "global_incidence": "~2.2M new cases/year (2024)",
        "mortality": "~1.8M deaths/year",
        "5yr_survival": "Stage I: 65%, Stage IV: 8%",
        "major_mutations": ["EGFR (15-20%)", "KRAS (25-30%)", "ALK (3-5%)",
                           "ROS1 (1-2%)", "BRAF (1-3%)", "MET exon 14 (3%)", "RET (1-2%)"],
        "approved_drugs_count": 45,
        "drug_classes": ["TKIs (1st/2nd/3rd gen)", "Immune checkpoint inhibitors",
                        "ADCs", "Bispecific antibodies", "Chemotherapy"],
        "key_drugs": [
            {"name": "Osimertinib", "target": "EGFR", "setting": "1L EGFRm ± adjuvant"},
            {"name": "Pembrolizumab", "target": "PD-1", "setting": "1L PD-L1 ≥50% ± chemo"},
            {"name": "Amivantamab", "target": "EGFR/MET", "setting": "ex20ins"},
            {"name": "Sotorasib", "target": "KRAS G12C", "setting": "2L+"},
            {"name": "Lorlatinib", "target": "ALK", "setting": "1L ALK+"},
        ],
        "market_size": "$32B (2024), projected $48B by 2030",
        "pipeline": {"phase_3": 85, "phase_2": 150, "phase_1": 100},
        "key_trends": ["Perioperative immunotherapy", "MRD-guided adjuvant therapy",
                       "Antibody-drug conjugates expanding", "Bispecifics entering 1L"],
    },
    "Breast Cancer": {
        "name": "Breast Cancer",
        "global_incidence": "~2.3M new cases/year (2024)",
        "mortality": "~685K deaths/year",
        "subtypes": ["HR+/HER2- (70%)", "HER2+ (15-20%)", "TNBC (10-15%)"],
        "approved_drugs_count": 52,
        "key_drugs": [
            {"name": "Trastuzumab deruxtecan", "target": "HER2", "setting": "HER2+ and HER2-low"},
            {"name": "Sacituzumab govitecan", "target": "Trop-2", "setting": "TNBC, HR+/HER2-"},
            {"name": "Palbociclib", "target": "CDK4/6", "setting": "HR+/HER2- 1L"},
            {"name": "Olaparib", "target": "PARP", "setting": "BRCA1/2-mutated"},
        ],
        "market_size": "$28B (2024)",
        "key_trends": ["CDK4/6 moving to adjuvant", "ADCs dominating HER2 space",
                       "Immunotherapy for TNBC", "Oral SERDs entering market"],
    },
}

MOCK_DRUG_SEARCH = {
    # Returned by any drug search; keyed by target/disease
    "default": [
        {"name": "Osimertinib", "target": "EGFR", "type": "Small molecule",
         "highest_phase": "Approved", "first_approved": "2015-11-13",
         "indications": ["NSCLC"], "company": "AstraZeneca"},
        {"name": "Gefitinib", "target": "EGFR", "type": "Small molecule",
         "highest_phase": "Approved", "first_approved": "2003-05-05",
         "indications": ["NSCLC"], "company": "AstraZeneca"},
        {"name": "Cetuximab", "target": "EGFR", "type": "Monoclonal antibody",
         "highest_phase": "Approved", "first_approved": "2004-02-12",
         "indications": ["CRC", "HNSCC"], "company": "Merck KGaA / BMS"},
        {"name": "Trastuzumab deruxtecan", "target": "HER2", "type": "ADC",
         "highest_phase": "Approved", "first_approved": "2019-12-20",
         "indications": ["Breast Cancer", "Gastric Cancer", "NSCLC"],
         "company": "Daiichi Sankyo / AstraZeneca"},
        {"name": "Pembrolizumab", "target": "PD-1", "type": "Monoclonal antibody",
         "highest_phase": "Approved", "first_approved": "2014-09-04",
         "indications": ["Melanoma", "NSCLC", "HNSCC", "cHL"], "company": "Merck (MSD)"},
        {"name": "Amivantamab", "target": "EGFR/MET", "type": "Bispecific antibody",
         "highest_phase": "Approved", "first_approved": "2021-05-21",
         "indications": ["NSCLC (ex20ins)"], "company": "Janssen"},
        {"name": "Sotorasib", "target": "KRAS G12C", "type": "Small molecule",
         "highest_phase": "Approved", "first_approved": "2021-05-28",
         "indications": ["NSCLC (KRAS G12C)"], "company": "Amgen"},
        {"name": "Sacituzumab govitecan", "target": "Trop-2", "type": "ADC",
         "highest_phase": "Approved", "first_approved": "2020-04-22",
         "indications": ["TNBC", "HR+/HER2- Breast Cancer"], "company": "Gilead"},
    ]
}

# =============================================================================
# CSS — Product-grade styling
# =============================================================================

CUSTOM_CSS = """
/* ===== Design Tokens ===== */
:root {
    --navy-950: #020617;
    --navy-900: #0a1628;
    --navy-800: #122540;
    --navy-700: #1a3050;
    --navy-600: #1e3a5f;
    --teal-500: #0891b2;
    --teal-400: #06b6d4;
    --teal-50: #ecfeff;
    --emerald-500: #10b981;
    --emerald-50: #ecfdf5;
    --surface: #ffffff;
    --bg: #f8fafc;
    --bg-alt: #f1f5f9;
    --text: #0f172a;
    --text-secondary: #475569;
    --text-tertiary: #94a3b8;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    --radius-sm: 6px;
    --radius: 10px;
    --radius-lg: 14px;
    --radius-xl: 20px;
    --shadow-xs: 0 1px 2px rgba(15,23,42,0.04);
    --shadow-sm: 0 1px 3px rgba(15,23,42,0.06);
    --shadow: 0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
    --shadow-md: 0 4px 6px -1px rgba(15,23,42,0.08), 0 2px 4px -2px rgba(15,23,42,0.04);
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'SF Mono', 'JetBrains Mono', 'Fira Code', monospace;
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    max-width: 100% !important;
    padding: 0 32px !important;
    margin: 0 auto !important;
    font-family: var(--font) !important;
    background: var(--bg) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Force ALL text dark — override Gradio Soft theme everywhere */
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .md,
.gradio-container .md *,
.gradio-container [class*="markdown"],
.gradio-container [class*="markdown"] * {
    color: var(--text) !important;
}
.gradio-container .prose h2, .gradio-container .md h2,
.gradio-container [class*="markdown"] h2 {
    color: var(--navy-950) !important;
    font-weight: 700 !important;
}
.gradio-container .prose strong, .gradio-container .md strong,
.gradio-container [class*="markdown"] strong {
    color: var(--navy-900) !important;
}
.gradio-container label, .gradio-container .label-text,
.gradio-container .label-container span {
    color: var(--text) !important;
    font-weight: 500 !important;
}
.gradio-container td { color: var(--text) !important; }
.gradio-container th { color: var(--text-secondary) !important; }

/* ===== Header ===== */
.header-container {
    background: linear-gradient(135deg, var(--navy-950) 0%, var(--navy-900) 30%, var(--navy-700) 70%, var(--navy-600) 100%);
    padding: 40px 48px 34px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.header-container::before {
    content: '';
    position: absolute;
    top: -120px;
    right: -80px;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle, rgba(6,182,212,0.13) 0%, rgba(6,182,212,0.04) 40%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.header-container::after {
    content: '';
    position: absolute;
    bottom: -60px;
    left: 30%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.header-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    position: relative;
    z-index: 1;
}
.header-title {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 10px 0;
    letter-spacing: -0.4px;
    line-height: 1.2;
    color: #ffffff;
}
.header-subtitle {
    font-size: 14px;
    opacity: 0.68;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
    max-width: 520px;
    color: rgba(255,255,255,0.85);
}
.header-badges {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid rgba(255,255,255,0.10);
    color: rgba(255,255,255,0.8);
}

/* ===== Tabs ===== */
.tabs { border: none !important; }
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 0 4px !important;
    box-shadow: none !important;
    margin-bottom: 24px !important;
    gap: 4px !important;
}
.tab-nav button {
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    border: none !important;
    color: var(--text-secondary) !important;
    transition: color 0.15s ease, background 0.15s ease !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    cursor: pointer !important;
}
.tab-nav button:hover {
    color: var(--text) !important;
    background: var(--bg-alt) !important;
}
.tab-nav button.selected {
    background: transparent !important;
    color: var(--navy-900) !important;
    border-bottom: 2px solid var(--navy-900) !important;
    box-shadow: none !important;
    font-weight: 600 !important;
}
.tab-nav button:focus-visible {
    outline: 2px solid var(--navy-600) !important;
    outline-offset: -2px !important;
    border-radius: 8px 8px 0 0 !important;
}

/* ===== Input ===== */
.agent-input textarea, .agent-input input {
    border-radius: var(--radius) !important;
    border: 1.5px solid var(--border) !important;
    padding: 14px 16px !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    resize: none !important;
    background: var(--surface) !important;
    color: var(--text) !important;
}
.agent-input textarea:focus, .agent-input input:focus {
    border-color: var(--navy-800) !important;
    box-shadow: 0 0 0 3px rgba(18,37,64,0.08) !important;
    outline: none !important;
}

/* Example chips */
.example-chip {
    border-radius: 20px !important;
    padding: 6px 16px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    min-width: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.example-chip:hover {
    border-color: var(--teal-400) !important;
    color: var(--teal-500) !important;
    background: var(--teal-50) !important;
}
.example-chip:focus-visible {
    outline: 2px solid var(--teal-400) !important;
    outline-offset: 1px !important;
}

/* Primary button */
.btn-primary {
    background: var(--navy-950) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 11px 32px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background 0.15s ease, box-shadow 0.15s ease !important;
    letter-spacing: 0.01em;
}
.btn-primary:hover {
    background: var(--navy-800) !important;
    box-shadow: var(--shadow-md);
}
.btn-primary:focus-visible {
    outline: 2px solid var(--navy-600) !important;
    outline-offset: 2px !important;
}

/* ===== Thinking Steps ===== */
.thinking-steps {
    background: linear-gradient(135deg, #f8faff, #f0fdfa);
    border: 1px solid #ccfbf1;
    border-radius: var(--radius);
    padding: 14px 18px;
    margin: 16px 0;
}
.thinking-step {
    padding: 3px 0;
    color: #115e59;
    font-size: 13px;
    line-height: 1.6;
}

/* ===== Cards ===== */
.card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: var(--shadow-xs);
    border: 1px solid var(--border);
    margin-bottom: 16px;
    transition: box-shadow 0.2s ease;
}
.card:hover { box-shadow: var(--shadow-sm); }

/* ===== Report Content ===== */
.report { color: var(--text); font-size: 14.5px; line-height: 1.75; }
.report h2 {
    font-size: 20px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 12px;
    color: var(--navy-950);
    letter-spacing: -0.3px;
}
.report h3 {
    font-size: 15px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 8px;
    color: var(--text);
}
.report table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 13.5px;
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--border);
}
.report th {
    background: var(--bg-alt);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 11.5px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
}
.report td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-light);
    color: var(--text);
}
.report tr:last-child td { border-bottom: none; }
.report tr:hover td { background: #fafcff; }

/* ===== Status Badge ===== */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-approved { background: #dcfce7; color: #166534; }
.badge-phase3  { background: #dbeafe; color: #1e40af; }
.badge-phase2  { background: #fef3c7; color: #92400e; }
.badge-phase1  { background: #fce7f3; color: #9d174d; }

/* ===== Footer ===== */
.footer {
    text-align: center;
    padding: 28px 16px;
    color: var(--text-tertiary);
    font-size: 12px;
    border-top: 1px solid var(--border-light);
    margin-top: 48px;
    letter-spacing: 0.02em;
}
.footer strong { color: var(--text-secondary); font-weight: 600; }

/* ===== Animations ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.35s ease-out; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    .fade-in { animation: none; }
}

/* Misc */
footer { display: none !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

# =============================================================================
# MCP CLIENT
# =============================================================================

_mcp_tools_cache: Optional[List[str]] = None

async def get_mcp_tools() -> List[str]:
    """Fetch available MCP tool names with caching."""
    global _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    if not API_KEY:
        return []
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(SERVER_URL, timeout=15, sse_read_timeout=15) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                _mcp_tools_cache = [t.name for t in result.tools]
                return _mcp_tools_cache
    except Exception as e:
        print(f"[MCP] Tool discovery failed: {e}")
        return []


async def call_mcp_tool(tool_name: str, args: dict) -> Optional[dict]:
    """Call a specific MCP tool. Returns parsed JSON or None on failure."""
    if not API_KEY:
        return None
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(SERVER_URL, timeout=30, sse_read_timeout=30) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=args)
                if result.content:
                    text = result.content[0].text
                    return json.loads(text) if isinstance(text, str) else text
    except Exception as e:
        print(f"[MCP] {tool_name} failed: {e}")
    return None


# =============================================================================
# AGENT ENGINE — Intent Parsing & Tool Routing
# =============================================================================

# Entity extraction patterns
TARGET_KEYWORDS = {
    "egfr": "EGFR", "her2": "HER2", "her-2": "HER2", "erbb2": "HER2",
    "pd-l1": "PD-L1", "pdl1": "PD-L1", "pd-1": "PD-1", "pd1": "PD-1",
    "braf": "BRAF", "alk": "ALK", "ros1": "ROS1", "kras": "KRAS",
    "vegf": "VEGF", "vegfr": "VEGFR", "ctla-4": "CTLA-4", "ctla4": "CTLA-4",
    "stat3": "STAT3", "met": "c-MET", "c-met": "c-MET", "ret": "RET",
    "ntrk": "NTRK", "fgfr": "FGFR", "parp": "PARP", "cdk4": "CDK4/6",
    "cdk4/6": "CDK4/6", "btk": "BTK", "jak": "JAK", "flt3": "FLT3",
    "idh1": "IDH1", "idh2": "IDH2", "tp53": "TP53", "p53": "TP53",
    "brca1": "BRCA1", "brca2": "BRCA2", "trop-2": "Trop-2", "trop2": "Trop-2",
    "claudin": "Claudin 18.2", "cldn18": "Claudin 18.2",
}

DISEASE_KEYWORDS = {
    "non-small cell lung cancer": "NSCLC", "nsclc": "NSCLC", "lung cancer": "Lung Cancer",
    "breast cancer": "Breast Cancer", "triple-negative breast": "Triple-Negative Breast Cancer",
    "tnbc": "Triple-Negative Breast Cancer",
    "colorectal": "Colorectal Cancer", "crc": "Colorectal Cancer",
    "melanoma": "Melanoma",
    "pancreatic": "Pancreatic Cancer",
    "hepatocellular": "Hepatocellular Carcinoma", "hcc": "Hepatocellular Carcinoma",
    "liver cancer": "Hepatocellular Carcinoma",
    "gastric": "Gastric Cancer", "stomach cancer": "Gastric Cancer",
    "leukemia": "Leukemia", "lymphoma": "Lymphoma",
    "ovarian": "Ovarian Cancer", "prostate": "Prostate Cancer",
    "multiple myeloma": "Multiple Myeloma", "mm": "Multiple Myeloma",
    "renal cell": "Renal Cell Carcinoma", "rcc": "Renal Cell Carcinoma",
    "bladder": "Urothelial Carcinoma", "urothelial": "Urothelial Carcinoma",
    "head and neck": "Head and Neck Cancer", "hnscc": "Head and Neck Cancer",
    "glioblastoma": "Glioblastoma", "gbm": "Glioblastoma",
    "alzheimer": "Alzheimer's Disease", "parkinson": "Parkinson's Disease",
    "diabetes": "Diabetes", "obesity": "Obesity",
}

COMPANY_KEYWORDS = {
    "roche": "Roche", "genentech": "Roche",
    "novartis": "Novartis",
    "pfizer": "Pfizer",
    "merck": "Merck (MSD)", "msd": "Merck (MSD)",
    "bristol-myers": "Bristol-Myers Squibb", "bms": "Bristol-Myers Squibb",
    "astrazeneca": "AstraZeneca", "az": "AstraZeneca",
    "johnson": "Johnson & Johnson", "jnj": "Johnson & Johnson", "janssen": "Johnson & Johnson",
    "sanofi": "Sanofi",
    "gsk": "GlaxoSmithKline",
    "abbvie": "AbbVie",
    "amgen": "Amgen",
    "gilead": "Gilead",
    "lilly": "Eli Lilly", "eli lilly": "Eli Lilly",
    "moderna": "Moderna",
    "biontech": "BioNTech",
    "daiichi": "Daiichi Sankyo",
    "beigene": "BeiGene", "百济": "BeiGene",
    "innovent": "Innovent", "信达": "Innovent",
    "hengrui": "Hengrui", "恒瑞": "Hengrui",
    "akebia": "Akebia",
}

PHASE_KEYWORDS = {
    "phase 3": "phase_3", "phase iii": "phase_3", "phase iii": "phase_3", "pivotal": "phase_3",
    "phase 2": "phase_2", "phase ii": "phase_2",
    "phase 1": "phase_1", "phase i": "phase_1", "first-in-human": "phase_1", "fih": "phase_1",
    "approved": "approved", "marketed": "approved", "launched": "approved",
    "preclinical": "preclinical",
}

MODULE_KEYWORDS = {
    "target": ["target", "靶点", "receptor", "kinase", "protein", "gene", "mutation", "pathway",
               "inhibitor target", "antibody target", "drug target"],
    "drug": ["drug", "药物", "medicine", "inhibitor", "antibody", "therapy", "treatment regimen",
             "approved drug", "pipeline drug", "molecule", "compound", "modality"],
    "disease": ["disease", "疾病", "cancer", "tumor", "indication", "适应症", "epidemiology",
                "patients", "prevalence", "incidence", "mortality"],
    "company": ["company", "公司", "pharma", "biotech", "pipeline of", "portfolio",
                "acquisition", "merger", "partner", "revenue"],
    "trial": ["trial", "试验", "clinical", "nct", "enrollment", "endpoint", "randomized",
              "double-blind", "phase 3 trial", "phase 2 trial", "phase 1 trial"],
}


def parse_intent(query: str) -> Dict:
    """
    Parse a natural language query to determine:
    - module (target/drug/disease/company/trial)
    - entities (targets, diseases, companies, phases)
    - mcp_args (for direct MCP call)
    - confidence
    """
    lower = query.lower()
    result = {
        "module": "target",  # default
        "entities": {"targets": [], "diseases": [], "companies": [], "phases": []},
        "mcp_args": {"limit": 10},
        "confidence": 0.0,
        "thinking": [],
    }

    # Extract entities
    for kw, val in TARGET_KEYWORDS.items():
        if kw in lower and val not in result["entities"]["targets"]:
            result["entities"]["targets"].append(val)

    for kw, val in DISEASE_KEYWORDS.items():
        if kw in lower and val not in result["entities"]["diseases"]:
            result["entities"]["diseases"].append(val)

    for kw, val in COMPANY_KEYWORDS.items():
        if kw in lower and val not in result["entities"]["companies"]:
            result["entities"]["companies"].append(val)

    for kw, val in PHASE_KEYWORDS.items():
        if kw in lower:
            result["entities"]["phases"].append(val)

    # Determine module by scoring keyword matches
    scores = {m: 0 for m in MODULE_KEYWORDS}
    for module, keywords in MODULE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[module] += 1

    best_module = max(scores, key=scores.get)
    max_score = scores[best_module]

    # Heuristic overrides based on entities found
    if result["entities"]["targets"] and not result["entities"]["diseases"] and not result["entities"]["companies"]:
        if max_score == 0 or best_module == "drug":
            result["module"] = "target" if scores["target"] >= scores["drug"] else "drug"
        else:
            result["module"] = best_module
    elif result["entities"]["diseases"] and not result["entities"]["targets"]:
        result["module"] = "disease"
    elif result["entities"]["companies"] and not result["entities"]["targets"]:
        result["module"] = "company"
    elif "trial" in lower or "clinical" in lower or "enrollment" in lower:
        result["module"] = "trial"
    else:
        result["module"] = best_module if max_score > 0 else "target"

    # Build MCP args based on module
    mod = result["module"]
    if mod == "target" and result["entities"]["targets"]:
        result["mcp_args"] = {"target": result["entities"]["targets"][:3], "limit": 10}
        result["thinking"].append(f"🔍 Detected target query → searching for: {', '.join(result['entities']['targets'][:3])}")
    elif mod == "drug":
        args = {"limit": 10}
        if result["entities"]["targets"]:
            args["target"] = result["entities"]["targets"][:3]
        if result["entities"]["diseases"]:
            args["disease"] = result["entities"]["diseases"][:3]
        if result["entities"]["phases"]:
            args["highest_phase"] = result["entities"]["phases"][:3]
        result["mcp_args"] = args
        result["thinking"].append(f"🔍 Detected drug query → searching with {json.dumps(args)}")
    elif mod == "disease" and result["entities"]["diseases"]:
        result["mcp_args"] = {"disease": result["entities"]["diseases"][:3], "limit": 10}
        result["thinking"].append(f"🔍 Detected disease query → searching for: {', '.join(result['entities']['diseases'][:3])}")
    elif mod == "company" and result["entities"]["companies"]:
        result["mcp_args"] = {"company": result["entities"]["companies"][:3], "limit": 10}
        result["thinking"].append(f"🔍 Detected company query → searching for: {', '.join(result['entities']['companies'][:3])}")
    elif mod == "trial":
        args = {"limit": 10}
        if result["entities"]["targets"]:
            args["target"] = result["entities"]["targets"][:3]
        if result["entities"]["diseases"]:
            args["disease"] = result["entities"]["diseases"][:3]
        if result["entities"]["phases"]:
            args["phase"] = result["entities"]["phases"][:3]
        result["mcp_args"] = args
        result["thinking"].append(f"🔍 Detected trial query → searching with {json.dumps(args)}")

    # Confidence
    entity_count = sum(len(v) for v in result["entities"].values())
    result["confidence"] = min(entity_count * 0.25 + scores[result["module"]] * 0.15, 0.95)

    return result


# =============================================================================
# REPORT BUILDERS
# =============================================================================

def _badge(phase: str) -> str:
    """Generate an HTML badge for a drug phase."""
    phase_lower = phase.lower()
    if "approved" in phase_lower:
        cls = "badge-approved"
    elif "phase 3" in phase_lower or "phase iii" in phase_lower:
        cls = "badge-phase3"
    elif "phase 2" in phase_lower or "phase ii" in phase_lower:
        cls = "badge-phase2"
    elif "phase 1" in phase_lower or "phase i" in phase_lower:
        cls = "badge-phase1"
    else:
        cls = "badge-phase2"
    return f'<span class="badge {cls}">{phase}</span>'


def build_target_report(target_data: Dict) -> str:
    """Build a structured target intelligence report."""
    name = target_data.get("name", "Unknown")
    full = target_data.get("full_name", "")
    family = target_data.get("family", "N/A")
    cls = target_data.get("class", "N/A")
    drugs = target_data.get("approved_drugs", [])
    pipeline = target_data.get("pipeline_summary", {})
    pathways = target_data.get("pathways", [])
    mutations = target_data.get("mutation_hotspots", [])
    landscape = target_data.get("competitive_landscape", "")

    report = []
    report.append(f"## 🎯 {name} — Target Intelligence Report")
    report.append("")

    # Overview card
    report.append("### 📋 Overview")
    report.append(f"| Property | Value |")
    report.append(f"|----------|-------|")
    report.append(f"| **Full Name** | {full} |")
    report.append(f"| **Family** | {family} |")
    report.append(f"| **Class** | {cls} |")
    report.append(f"| **Approved Drugs** | {len(drugs)} |")
    report.append("")

    # Pathways
    if pathways:
        report.append("### 🧬 Signaling Pathways")
        for p in pathways:
            report.append(f"- {p}")
        report.append("")

    # Mutations
    if mutations:
        report.append("### 🔬 Key Mutations / Variants")
        for m in mutations:
            report.append(f"- {m}")
        report.append("")

    # Approved Drugs Table
    if drugs:
        report.append(f"### 💊 Approved Drugs ({len(drugs)})")
        report.append("| Drug | Type | Generation | Year | Indications | Company |")
        report.append("|------|------|-----------|------|-------------|---------|")
        for d in drugs:
            inds = ", ".join(d.get("indications", [])[:2])
            if len(d.get("indications", [])) > 2:
                inds += f" +{len(d['indications']) - 2} more"
            report.append(f"| {d['name']} | {d['type']} | {d.get('gen', '-')} | "
                         f"{d['year']} | {inds} | {d.get('company', '-')} |")
        report.append("")

    # Pipeline
    if pipeline:
        report.append("### 🔬 Pipeline Overview")
        report.append(f"| Phase | Count |")
        report.append(f"|-------|-------|")
        for phase in ["phase_3", "phase_2", "phase_1", "preclinical"]:
            label = phase.replace("_", " ").title()
            report.append(f"| {label} | {pipeline.get(phase, 'N/A')} |")
        report.append("")
        hot = pipeline.get("hot_topics", [])
        if hot:
            report.append("**🔥 Hot Topics:**")
            for t in hot:
                report.append(f"- {t}")
            report.append("")

    # Competitive landscape
    if landscape:
        report.append("### 🏔️ Competitive Landscape")
        report.append(landscape)
        report.append("")

    report.append("---")
    report.append(f"*Report generated by PatSnap Pharma Intelligence Agent*")
    return "\n".join(report)


def build_drug_report(items: List[Dict], query_summary: str, total: int = 0) -> str:
    """Build a drug pipeline report from search results."""
    if not items:
        return f"## 💊 Drug Search: {query_summary}\n\n📭 No results found. Try a different query."

    drug_types = Counter()
    companies = Counter()
    years = []
    phases = Counter()

    for item in items:
        drug_types[item.get("type", "Unknown")] += 1
        companies[item.get("company", "Unknown")] += 1
        phases[item.get("highest_phase", "Unknown")] += 1
        date = item.get("first_approved", "")
        if date and date != "N/A":
            try:
                years.append(int(date.split("-")[0]))
            except (ValueError, IndexError):
                pass

    report = []
    report.append(f"## 💊 {query_summary}")
    report.append(f"*{len(items)} results{' of ' + str(total) if total else ''}*")
    report.append("")

    # Stats row
    report.append("### 📊 Summary")
    stats = []
    if drug_types:
        top = drug_types.most_common(3)
        stats.append(f"**Top Types:** " + " · ".join(f"{t} ({c})" for t, c in top))
    if phases:
        top_p = phases.most_common(3)
        stats.append(f"**Phases:** " + " · ".join(f"{p} ({c})" for p, c in top_p))
    if years:
        stats.append(f"**Timeline:** {min(years)} – {max(years)}")
    if companies:
        top_c = companies.most_common(3)
        stats.append(f"**Top Companies:** " + " · ".join(f"{c} ({n})" for c, n in top_c))
    for s in stats:
        report.append(f"- {s}")
    report.append("")

    # Drug table
    report.append("### 🧪 Pipeline Details")
    report.append("| Drug | Type | Phase | Indications | Company |")
    report.append("|------|------|-------|-------------|---------|")
    for item in items[:15]:
        name = item.get("name", "N/A")
        dtype = item.get("type", "N/A")
        phase = item.get("highest_phase", "N/A")
        inds = ", ".join(item.get("indications", ["N/A"])[:2])
        comp = item.get("company", "N/A")
        report.append(f"| {name} | {dtype} | {_badge(phase)} | {inds} | {comp} |")
    report.append("")

    report.append("---")
    report.append(f"*Report generated by PatSnap Pharma Intelligence Agent*")
    return "\n".join(report)


def build_disease_report(disease_data: Dict) -> str:
    """Build a disease landscape report."""
    name = disease_data.get("name", "Unknown")
    report = []
    report.append(f"## 🏥 {name} — Disease Landscape")
    report.append("")

    report.append("### 📊 Epidemiology")
    for key in ["global_incidence", "mortality", "5yr_survival"]:
        if key in disease_data:
            label = key.replace("5yr", "5-Year ").replace("_", " ").title()
            report.append(f"- **{label}:** {disease_data[key]}")
    report.append("")

    mutations = disease_data.get("major_mutations", [])
    if mutations:
        report.append("### 🧬 Key Driver Mutations")
        for m in mutations:
            report.append(f"- {m}")
        report.append("")

    key_drugs = disease_data.get("key_drugs", [])
    if key_drugs:
        report.append("### 💊 Key Therapies")
        report.append("| Drug | Target | Setting |")
        report.append("|------|--------|---------|")
        for d in key_drugs:
            report.append(f"| {d['name']} | {d['target']} | {d.get('setting', '-')} |")
        report.append("")

    report.append("### 📈 Market & Pipeline")
    for key in ["market_size", "approved_drugs_count"]:
        if key in disease_data:
            label = key.replace("_", " ").title().replace("Drugs", "Drugs").replace("Count", "Count")
            report.append(f"- **{label}:** {disease_data[key]}")
    pipeline = disease_data.get("pipeline", {})
    if pipeline:
        report.append(f"- **Pipeline:** Phase 3: {pipeline.get('phase_3', 'N/A')} | "
                     f"Phase 2: {pipeline.get('phase_2', 'N/A')} | Phase 1: {pipeline.get('phase_1', 'N/A')}")
    report.append("")

    trends = disease_data.get("key_trends", [])
    if trends:
        report.append("### 🔥 Key Trends")
        for t in trends:
            report.append(f"- {t}")
        report.append("")

    report.append("---")
    report.append(f"*Report generated by PatSnap Pharma Intelligence Agent*")
    return "\n".join(report)


def build_company_report(company_data: Dict) -> str:
    """Build a company profile report."""
    name = company_data.get("name", "Unknown")
    report = []
    report.append(f"## 🏢 {name} — Company Profile")
    report.append("")

    report.append("### 📋 Company Overview")
    for key in ["ticker", "headquarters", "employees", "2024_revenue"]:
        if key in company_data:
            label = key.replace("_", " ").title().replace("2024", "2024")
            report.append(f"- **{label}:** {company_data[key]}")
    report.append("")

    ta = company_data.get("therapeutic_areas", [])
    if ta:
        report.append(f"**Therapeutic Areas:** " + " · ".join(ta))
        report.append("")

    flagship = company_data.get("flagship_drugs", [])
    if flagship:
        report.append("### 💊 Flagship Drugs")
        report.append("| Drug | Target | Sales | Phase |")
        report.append("|------|--------|-------|-------|")
        for d in flagship:
            report.append(f"| {d['name']} | {d['target']} | {d['sales']} | {_badge(d['phase'])} |")
        report.append("")

    pipeline = company_data.get("pipeline_count", {})
    if pipeline:
        report.append("### 🔬 Pipeline Overview")
        report.append(f"| Phase | Count |")
        report.append(f"|-------|-------|")
        for phase in ["approved", "phase_3", "phase_2", "phase_1"]:
            label = phase.replace("_", " ").title()
            report.append(f"| {label} | {pipeline.get(phase, 'N/A')} |")
        report.append("")

    deals = company_data.get("recent_deals", [])
    if deals:
        report.append("### 🤝 Recent Strategic Deals")
        for d in deals:
            report.append(f"- {d}")
        report.append("")

    strategy = company_data.get("strategy", "")
    if strategy:
        report.append(f"### 🎯 Strategic Outlook\n{strategy}")
        report.append("")

    report.append("---")
    report.append(f"*Report generated by PatSnap Pharma Intelligence Agent*")
    return "\n".join(report)


def build_agent_thinking(intent: Dict, mcp_available: bool, mcp_result: Optional[Dict]) -> str:
    """Build the agent thinking trace HTML."""
    steps = []
    steps.append(f"🧠 **Intent:** {intent['module'].title()} query (confidence: {intent['confidence']:.0%})")

    if intent["entities"]["targets"]:
        steps.append(f"🎯 **Targets:** {', '.join(intent['entities']['targets'])}")
    if intent["entities"]["diseases"]:
        steps.append(f"🏥 **Diseases:** {', '.join(intent['entities']['diseases'])}")
    if intent["entities"]["companies"]:
        steps.append(f"🏢 **Companies:** {', '.join(intent['entities']['companies'])}")

    for think in intent.get("thinking", []):
        steps.append(think)

    if mcp_available:
        if mcp_result:
            total = mcp_result.get("total", 0)
            items = len(mcp_result.get("items", []))
            steps.append(f"✅ **MCP Live Data:** Retrieved {items} of {total} records")
        else:
            steps.append(f"⚠️ **MCP:** No results from live API, using knowledge base")
    else:
        steps.append(f"📚 **Source:** Knowledge base (no API key configured)")

    return "\n".join(f"<div class='thinking-step'>{s}</div>" for s in steps)


# =============================================================================
# CORE AGENT EXECUTION
# =============================================================================

async def run_agent(query: str, lang: str = "en") -> Tuple[str, str]:
    """
    Execute the full agent pipeline:
    1. Parse intent
    2. Try MCP → fallback to knowledge base
    3. Build report

    Returns (thinking_html, report_markdown)
    """
    if not query or not query.strip():
        return "", "*👋 Welcome! Ask me anything about drug targets, diseases, companies, or clinical trials.*"

    # Step 1: Parse intent
    intent = parse_intent(query)
    module = intent["module"]
    mod_info = MODULES[module]

    # Step 2: Try MCP
    mcp_data = None
    mcp_available = bool(API_KEY)
    if mcp_available:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
            async with streamablehttp_client(SERVER_URL, timeout=25, sse_read_timeout=25) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool_names = [t.name for t in tools.tools]

                    # Find matching tool
                    tool_to_call = None
                    for t in [mod_info["tool"], "ls_drug_search"]:  # fallback to drug search
                        if t in tool_names:
                            tool_to_call = t
                            break

                    if tool_to_call:
                        result = await session.call_tool(tool_to_call, arguments=intent["mcp_args"])
                        if result.content:
                            text = result.content[0].text
                            mcp_data = json.loads(text) if isinstance(text, str) else text
        except Exception as e:
            print(f"[Agent] MCP error: {e}")

    # Step 3: Build thinking trace
    if module == "target" and intent["entities"]["targets"]:
        entity_name = intent["entities"]["targets"][0]
    elif module == "disease" and intent["entities"]["diseases"]:
        entity_name = intent["entities"]["diseases"][0]
    elif module == "company" and intent["entities"]["companies"]:
        entity_name = intent["entities"]["companies"][0]
    else:
        entity_name = query[:50]

    thinking = build_agent_thinking(intent, mcp_available, mcp_data)

    # Step 4: Build report
    report = ""

    if mcp_data and mcp_data.get("items"):
        # Live MCP data
        items = mcp_data["items"]
        total = mcp_data.get("total", len(items))
        report = build_drug_report(items, f"Results for \"{entity_name}\"", total)
    elif module == "target":
        # Knowledge base fallback for targets
        target_key = entity_name.upper()
        target_data = MOCK_TARGETS.get(target_key)
        if target_data:
            report = build_target_report(target_data)
        else:
            report = build_drug_report(
                MOCK_DRUG_SEARCH["default"],
                f"Results for \"{entity_name}\" (target overview)",
                len(MOCK_DRUG_SEARCH["default"])
            )
    elif module == "disease":
        # Knowledge base fallback for diseases
        disease_key = None
        for kw, val in DISEASE_KEYWORDS.items():
            if val.upper() == entity_name.upper():
                disease_key = val
                break
        if not disease_key:
            disease_key = entity_name
        disease_data = MOCK_DISEASES.get(disease_key)
        if disease_data:
            report = build_disease_report(disease_data)
        else:
            report = build_drug_report(
                MOCK_DRUG_SEARCH["default"],
                f"Results for \"{entity_name}\" (disease overview)",
            )
    elif module == "company":
        # Knowledge base fallback for companies
        company_data = MOCK_COMPANIES.get(entity_name)
        if company_data:
            report = build_company_report(company_data)
        else:
            report = f"## 🏢 {entity_name}\n\n*Detailed company profile not available in demo mode. "
            report += "Connect a PatSnap API key for live data.*"
    else:
        # Default drug report from knowledge base
        report = build_drug_report(
            MOCK_DRUG_SEARCH["default"],
            f"Results for \"{entity_name}\"",
        )

    return thinking, report


# =============================================================================
# GRADIO UI
# =============================================================================

def create_header():
    """Create the app header block."""
    return gr.HTML("""
    <div class="header-container">
        <div class="header-row">
            <div>
                <h1 class="header-title">PatSnap Pharma Intelligence</h1>
                <p class="header-subtitle">
                    AI-powered drug discovery intelligence. Explore targets, pipelines, diseases, companies, and clinical trials — all in one place.
                </p>
            </div>
        </div>
        <div class="header-badges">
            <span class="header-badge">Multi-Module Agent</span>
            <span class="header-badge">MCP Live Data</span>
            <span class="header-badge">Structured Reports</span>
        </div>
    </div>
    """)


def create_footer():
    """Create the app footer."""
    return gr.HTML("""
    <div class="footer">
        <strong>PatSnap Pharma Intelligence</strong> — Powered by PatSnap Life Sciences MCP<br>
        <span style="opacity:0.7">Data sourced from PatSnap's pharmaceutical intelligence platform. Demo data shown where API key is not configured.</span>
    </div>
    """)


def build_app():
    """Build the complete Gradio application."""

    with gr.Blocks(
        title="PatSnap Pharma Intelligence",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="emerald",
            neutral_hue="slate",
        ).set(
            body_text_color="*neutral_900",
            body_text_color_subdued="*neutral_800",
            body_text_color_dark="*neutral_900",
            block_title_text_color="*neutral_900",
            block_title_text_color_dark="*neutral_200",
            block_label_text_color="*neutral_800",
            block_label_text_color_dark="*neutral_300",
            input_text_color="*neutral_900",
            link_text_color="*primary_600",
            link_text_color_dark="*primary_300",
            table_odd_background_fill="*neutral_50",
            table_even_background_fill="*neutral_100",
        ),
        analytics_enabled=False,
    ) as app:
        create_header()

        # ===== TABS =====
        with gr.Tabs(elem_classes="tabs"):

            # ========== TAB 1: AGENT CHAT (Overview) ==========
            with gr.TabItem("Agent", id="chat"):
                # Input area
                with gr.Group(elem_classes="card"):
                    chat_input = gr.Textbox(
                        label="Ask anything about drug discovery",
                        placeholder="e.g. Analyze EGFR's competitive landscape and pipeline drugs...",
                        lines=2,
                        elem_classes="agent-input",
                        show_label=True,
                    )
                    with gr.Row():
                        chat_btn = gr.Button("Analyze", variant="primary", elem_classes="btn-primary")
                        clear_btn = gr.Button("Clear", variant="secondary", size="sm")

                # Example chips
                example_btns = []
                examples = [
                    ("EGFR Analysis", "Analyze EGFR as a drug target — approved drugs, pipeline, and competitive landscape"),
                    ("PD-1 Drugs", "What drugs target PD-1? Show me approved and pipeline drugs"),
                    ("NSCLC Overview", "Give me a disease overview for non-small cell lung cancer"),
                    ("Roche Profile", "Profile Roche's oncology pipeline and recent deals"),
                    ("ALK Trials", "What clinical trials are targeting ALK in NSCLC?"),
                    ("HER2 ADC", "Show me HER2-targeting antibody-drug conjugates"),
                ]
                with gr.Row():
                    for label, prompt in examples:
                        btn = gr.Button(label, elem_classes="example-chip", size="sm")
                        example_btns.append((btn, prompt))

                # Agent thinking
                thinking_display = gr.HTML(
                    value="<div class='thinking-steps'><div class='thinking-step'>Agent ready. Ask a question to begin.</div></div>",
                    visible=True,
                )

                # Report output
                report_display = gr.Markdown(
                    value="## Welcome\n\n"
                          "I analyze drug targets, pipelines, diseases, companies, and clinical trials. "
                          "Just describe what you're looking for — the agent will understand your intent, "
                          "search relevant data sources, and generate a structured report.\n\n"
                          "| Module | What it does |\n"
                          "|--------|-------------|\n"
                          "| 🎯 **Target Intelligence** | Deep analysis of drug targets |\n"
                          "| 💊 **Drug Exploration** | Pipeline drugs by target, disease, or company |\n"
                          "| 🏥 **Disease Investigation** | Disease landscape & treatment overview |\n"
                          "| 🏢 **Company Profiling** | Pharma pipeline & strategic analysis |\n"
                          "| 🧪 **Clinical Trials** | Trial landscape by indication |\n\n"
                          "*Try a quick example above, or type your own question.*",
                    elem_classes="report",
                )

                # Wire up agent chat
                async def handle_chat(query):
                    thinking, report = await run_agent(query)
                    thinking_html = f"<div class='thinking-steps'>{thinking}</div>"
                    return thinking_html, report

                chat_btn.click(
                    fn=handle_chat,
                    inputs=[chat_input],
                    outputs=[thinking_display, report_display],
                )

                for btn, prompt in example_btns:
                    btn.click(
                        fn=lambda p=prompt: p,
                        outputs=[chat_input],
                    ).then(
                        fn=handle_chat,
                        inputs=[chat_input],
                        outputs=[thinking_display, report_display],
                    )

                clear_btn.click(
                    fn=lambda: ("", "", "## Welcome\n\n*Ready for your next query.*"),
                    outputs=[chat_input, thinking_display, report_display],
                )

            # ========== TAB 2-6: Structured Module Views ==========
            # These reuse the same agent but with module-specific context
            for mod_key, mod_info in MODULES.items():
                icon = mod_info["icon"]
                label = mod_info["label"]
                desc = mod_info["desc"]

                with gr.TabItem(f"{icon} {label}", id=mod_key):
                    gr.Markdown(f"### {icon} {label}\n{desc}", elem_classes="fade-in")

                    with gr.Row(elem_classes="card"):
                        with gr.Column(scale=3):
                            mod_input = gr.Textbox(
                                label=f"Search {mod_info['entity']}",
                                placeholder=f"Enter a {mod_info['entity']} name, e.g. EGFR, HER2, PD-L1...",
                                lines=1,
                                elem_classes="agent-input",
                            )
                            mod_btn = gr.Button(f"{icon} Search", variant="primary", elem_classes="btn-primary")

                    mod_thinking = gr.HTML(visible=True)
                    mod_report = gr.Markdown(
                        value=f"### {icon} {label}\n\n*Enter a {mod_info['entity']} above to generate a report.*",
                        elem_classes="report",
                    )

                    async def handle_module_search(q, mk=mod_key):
                        if not q.strip():
                            return "", f"### {MODULES[mk]['icon']} {MODULES[mk]['label']}\n\n*Please enter a search term.*"
                        # Route to the appropriate module by prefixing the query
                        routed_query = f"[{mk}] {q}"
                        thinking, report = await run_agent(routed_query)
                        thinking_html = f"<div class='thinking-steps'>{thinking}</div>"
                        return thinking_html, report

                    mod_btn.click(
                        fn=handle_module_search,
                        inputs=[mod_input],
                        outputs=[mod_thinking, mod_report],
                    )

                    # Pre-made reports
                    if mod_key == "target":
                        gr.Markdown("#### 📊 Featured Target Reports")
                        with gr.Row():
                            for target_name in ["EGFR", "HER2", "PD-L1"]:
                                td = MOCK_TARGETS.get(target_name)
                                if td:
                                    with gr.Column(scale=1):
                                        with gr.Group(elem_classes="card"):
                                            gr.Markdown(f"#### {target_name}\n{td.get('approved_drugs', [{}])[0].get('name', '')} ...")
                                            btn = gr.Button(f"View {target_name} Report →", size="sm")
                                            def show_report(tn=target_name):
                                                td = MOCK_TARGETS.get(tn)
                                                if td:
                                                    return build_target_report(td)
                                                return "Report not available."
                                            btn.click(fn=show_report, outputs=[mod_report])

                    elif mod_key == "company":
                        gr.Markdown("#### 📊 Featured Company Reports")
                        with gr.Row():
                            for cname in ["Roche", "AstraZeneca"]:
                                cd = MOCK_COMPANIES.get(cname)
                                if cd:
                                    with gr.Column(scale=1):
                                        with gr.Group(elem_classes="card"):
                                            gr.Markdown(f"#### {cname}\n{cd.get('headquarters', '')} — {cd.get('2024_revenue', '')}")
                                            btn = gr.Button(f"View {cname} Report →", size="sm")
                                            def show_crpt(cn=cname):
                                                cd = MOCK_COMPANIES.get(cn)
                                                if cd:
                                                    return build_company_report(cd)
                                                return "Report not available."
                                            btn.click(fn=show_crpt, outputs=[mod_report])

                    elif mod_key == "disease":
                        gr.Markdown("#### 📊 Featured Disease Reports")
                        with gr.Row():
                            for dname in ["NSCLC", "Breast Cancer"]:
                                dd = MOCK_DISEASES.get(dname)
                                if dd:
                                    with gr.Column(scale=1):
                                        with gr.Group(elem_classes="card"):
                                            gr.Markdown(f"#### {dname}\n{dd.get('global_incidence', '')}")
                                            btn = gr.Button(f"View {dname} Report →", size="sm")
                                            def show_drpt(dn=dname):
                                                dd = MOCK_DISEASES.get(dn)
                                                if dd:
                                                    return build_disease_report(dd)
                                                return "Report not available."
                                            btn.click(fn=show_drpt, outputs=[mod_report])

        # ===== FOOTER =====
        create_footer()

    return app


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=3, max_size=20)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
