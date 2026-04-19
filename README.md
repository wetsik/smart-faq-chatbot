Lead Scoring & Pipeline Manager
===============================

Streamlit dashboard redesigned from the old FAQ chatbot into Project 3 from the documentation:

- KPI cards for filtered leads, average score, high-priority leads, and total pipeline value
- sidebar filters for manager, stage, industry, conversion probability, and budget range
- sortable lead table for CRM review
- bar charts for pipeline stage distribution and manager performance
- Bitrix24-ready field support through `bitrix_owner` and CRM-style pipeline stages
- separate test CSV dataset in `sample_leads.csv`
- conversion probability and pricing recommendation for each lead

Run locally
-----------

```bash
pip install -r requirements.txt
streamlit run app.py
```

Project idea
------------

This app analyzes uploaded CRM leads or a built-in sample dataset, calculates a lead score, conversion probability, pricing recommendation, and presents the result in a dashboard. It follows the Project 3 brief:

- multiple KPI columns
- dataframe with sorting
- sidebar filters
- charts for distribution and performance
- expandable lead details

Recommended CSV columns
-----------------------

Use a CSV with fields like:

```text
lead_id,company,manager,industry,source,region,stage,budget,deal_value,last_activity_days,meetings_booked,email_open_rate,employees,bitrix_owner
```

Bitrix24 note
-------------

The current version is Bitrix24-ready at the UI/data level. If needed later, you can extend it with direct Bitrix24 API sync for leads, deals, and pipeline stages.

Deploy
------

The project is designed for Streamlit Cloud deployment and matches the documentation criteria:

- Streamlit app
- browser-accessible live URL after deployment
- public or private Streamlit Cloud access
