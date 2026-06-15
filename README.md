<div align="center">

# Census Data Management and Income Prediction System

### Full-stack PostgreSQL + Python + Streamlit + Heroku platform for demographic and income data management with ML income prediction

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Heroku](https://img.shields.io/badge/Heroku-430098?style=for-the-badge&logo=heroku&logoColor=white)](https://heroku.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

![Dataset Size](https://img.shields.io/badge/Dataset-215K_Records-blue?style=for-the-badge)

</div>

---

## Overview

This project implements a **full-stack data management and income prediction platform** built on a normalized **PostgreSQL relational database** containing 215,000 rows of US Census demographic and employment data. The system supports complete CRUD operations via REST API endpoints, delivers an interactive **Streamlit analytics dashboard**, and includes a **machine learning model** predicting whether an individual earns above $50K annually based on socioeconomic features.

The project demonstrates production-grade data engineering: BCNF-normalized schema design, Python-to-PostgreSQL bulk ingestion, server-side pagination and filtering, and cloud deployment to **Heroku**.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Census Data Management Platform                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Streamlit Frontend                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐ │   │
│  │  │ Dashboard │  │  Manage  │  │   Analytics Tool      │ │   │
│  │  │  KPIs    │  │   Data   │  │   Bar charts, Plots   │ │   │
│  │  │  Filters │  │  CRUD UI │  │   Income distribution │ │   │
│  │  └──────────┘  └──────────┘  └───────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────┘   │
│                              │ HTTP                             │
│  ┌───────────────────────────▼─────────────────────────────┐   │
│  │                Python API Layer                          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  CRUD Endpoints (psycopg2 + SQLAlchemy)          │   │   │
│  │  │  Create · Read · Update · Delete                 │   │   │
│  │  │  Pagination · Filtering · Sorting               │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └───────────────────────────┬─────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────▼─────────────────────────────┐   │
│  │             PostgreSQL Database (BCNF Normalized)         │   │
│  │  Individuals · Employment · JobDetails · Education       │   │
│  │  RelationshipDetails · IncomeDetails                     │   │
│  │  215,000 rows · Foreign key integrity                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            ML Income Predictor                          │   │
│  │  Logistic Regression / Random Forest                    │   │
│  │  Target: Income > $50K vs. <= $50K                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Deployed on Heroku (cloud PaaS)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema — BCNF Normalized

The schema is normalized to **Boyce-Codd Normal Form (BCNF)** to eliminate data redundancy and ensure referential integrity across all 215,000 records.

### Entity Relationship Overview

```
Individuals (individual_id PK)
    │
    ├──► Employment (individual_id FK)
    │       capital_gain, capital_loss, hours_per_week
    │
    ├──► JobDetails (individual_id FK)
    │       workclass, occupation, native_country
    │
    ├──► EducationDetails (individual_id FK)
    │       education, education_num
    │
    ├──► RelationshipDetails (individual_id FK)
    │       marital_status, relationship, sex
    │
    └──► IncomeDetails (individual_id FK)
            income (>50K / <=50K)
```

### Table Definitions

**Individuals** — Core demographic record
```sql
CREATE TABLE Individuals (
    individual_id   SERIAL PRIMARY KEY,
    age             INTEGER NOT NULL,
    race            VARCHAR(50),
    fnlwgt          INTEGER          -- Final weight (census sampling)
);
```

**EducationDetails** — Normalized education data
```sql
CREATE TABLE EducationDetails (
    edu_id          SERIAL PRIMARY KEY,
    individual_id   INTEGER REFERENCES Individuals(individual_id),
    education       VARCHAR(50) NOT NULL,
    education_num   INTEGER NOT NULL
);
```

**IncomeDetails** — Target variable table
```sql
CREATE TABLE IncomeDetails (
    income_id       SERIAL PRIMARY KEY,
    individual_id   INTEGER REFERENCES Individuals(individual_id),
    income          VARCHAR(10) NOT NULL  -- '>50K' or '<=50K'
);
```

---

## Dataset

| Attribute | Value |
|---|---|
| **Source** | UCI Census Income Dataset |
| **Total Records** | 215,000 rows |
| **Original Features** | 14 (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country) |
| **Target** | Income: >$50K or ≤$50K |
| **Normalization** | BCNF across 6 tables |
| **Ingestion Method** | Python bulk insert via psycopg2 |

---

## CRUD API Operations

The Python API layer exposes the following operations against the PostgreSQL database:

| Operation | Method | Description |
|---|---|---|
| **Create** | INSERT | Add new demographic records to relevant normalized tables |
| **Read** | SELECT | Retrieve income classifications by age, occupation, education, etc. |
| **Update** | UPDATE | Modify existing records (employment, job details, income) |
| **Delete** | DELETE | Remove records from non-constrained tables (Individuals deletion restricted by FK) |

### API Design Patterns

```python
# Server-side pagination for large datasets
def get_individuals(page: int, page_size: int = 1000):
    offset = (page - 1) * page_size
    query = """
        SELECT i.individual_id, i.age, e.education, inc.income
        FROM Individuals i
        JOIN EducationDetails e ON i.individual_id = e.individual_id
        JOIN IncomeDetails inc ON i.individual_id = inc.individual_id
        LIMIT %s OFFSET %s
    """
    return cursor.execute(query, (page_size, offset))

# Analytical aggregation query
def get_income_by_occupation():
    query = """
        SELECT jd.occupation,
               COUNT(*) as total,
               SUM(CASE WHEN inc.income = '>50K' THEN 1 ELSE 0 END) as high_earners,
               ROUND(AVG(i.age), 1) as avg_age
        FROM Individuals i
        JOIN JobDetails jd ON i.individual_id = jd.individual_id
        JOIN IncomeDetails inc ON i.individual_id = inc.individual_id
        GROUP BY jd.occupation
        ORDER BY high_earners DESC
    """
```

---

## ML Income Predictor

A supervised classification model predicts whether an individual earns above or below $50K annually based on their demographic and employment profile.

| Attribute | Details |
|---|---|
| **Task** | Binary classification (>50K vs. ≤50K) |
| **Features** | Age, education, occupation, marital status, hours per week, capital gain/loss |
| **Model** | Logistic Regression / Random Forest |
| **Library** | scikit-learn |
| **Integration** | Streamlit dashboard prediction interface |

---

## Streamlit Dashboard Features

### Dashboard Tab
- Total records count, income distribution KPIs
- - Adjustable record display (5,000 – 50,000 rows)
  - - CSV export functionality
    - - Income breakdown by demographic filters
     
      - ### Manage Data Tab
      - - Full CRUD UI for all 6 normalized tables
        - - Form-based record creation with validation
          - - Edit and delete individual records
            - - Real-time database reflection
             
              - ### Analytics Tool Tab
              - - Income distribution by occupation (bar chart)
                - - Education level vs. income comparison
                  - - Age distribution histogram by income class
                    - - Gender and marital status income analysis
                     
                      - ---

                      ## Installation and Setup

                      ### Prerequisites

                      ```bash
                      Python >= 3.8
                      PostgreSQL >= 13
                      pip install -r requirements.txt
                      ```

                      ### Core Dependencies

                      ```bash
                      pip install psycopg2-binary SQLAlchemy streamlit pandas scikit-learn
                      ```

                      ### Database Setup

                      ```sql
                      -- Create database
                      CREATE DATABASE census_data;

                      -- Run schema creation script
                      \i schema/create_tables.sql

                      -- Load data (Python ingestion script)
                      python ingest/load_census_data.py
                      ```

                      ### Run Locally

                      ```bash
                      # Start the Streamlit application
                      streamlit run app.py
                      # → Opens at http://localhost:8501
                      ```

                      ### Deploy to Heroku

                      ```bash
                      # Login and create app
                      heroku login
                      heroku create census-data-app

                      # Provision PostgreSQL addon
                      heroku addons:create heroku-postgresql:mini

                      # Set environment variables
                      heroku config:set DB_URL=$(heroku config:get DATABASE_URL)

                      # Deploy
                      git push heroku main
                      ```

                      ---

                      ## Why Relational Over Flat Files?

                      | Concern | Flat Files (CSV/Excel) | PostgreSQL |
                      |---|---|---|
                      | Scalability | Limited (Excel: 1M rows max) | Millions of rows with indexing |
                      | Data Integrity | No constraints | FK, PK, NOT NULL, CHECK enforced |
                      | Concurrency | Single user | Multi-user concurrent access |
                      | Query Complexity | Manual joins, limited aggregation | Full SQL: JOINs, subqueries, window functions |
                      | ACID Compliance | None | Full transaction safety |

                      ---

                      ## License

                      MIT License — open for data engineering research and adaptation.

                      ---

                      <div align="center">

                      Built by [Aditya Srivatsav Lolla](https://www.linkedin.com/in/lolla-aditya-srivatsav-2296671b0/) | MS Data Science, SUNY University at Buffalo

                      *"Production-grade data engineering: normalized schemas, CRUD APIs, and ML pipelines on 215K census records."*

                      </div>
