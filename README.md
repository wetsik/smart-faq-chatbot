Smart FAQ Chatbot (RAG)
=======================

Streamlit chatbot for FAQ / knowledge-base Q&A with:

- file upload for PDF, TXT, MD, and CSV knowledge sources
- semantic retrieval with TF-IDF cosine similarity
- chat UI with `st.chat_message` and `st.chat_input`
- sidebar conversation history
- progress indicator while indexing
- session-state powered multi-turn conversations

Run locally
-----------

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional healthcheck endpoint
-----------------------------

This project can also start a tiny HTTP healthcheck server, similar to a keep-alive `/health` route.

Public app URL:

- `https://smart-faq-chatbot-qgjc2afkvrqdothrrasrs2.streamlit.app/`

Set a separate port before startup:

```bash
HEALTHCHECK_ENABLED=true
HEALTHCHECK_PORT=8080
streamlit run app.py
```

If your platform exposes that extra port, your monitor can ping:

- `http://your-host:8080/health`
- `http://your-host:8080/`

Important:

- `HEALTHCHECK_PORT` must be different from the Streamlit port.
- If your hosting platform exposes only the main Streamlit port, this workaround may not be reachable from the outside.
- On platforms that force app sleeping at the infrastructure level, `/health` helps only if the host actually counts that ping as external traffic.
- For this Streamlit deployment, point the external monitor to `https://smart-faq-chatbot-qgjc2afkvrqdothrrasrs2.streamlit.app/`.

How it works
------------

1. Keep `Use sample FAQ` enabled for a ready demo, or upload your own FAQ files.
2. Click `Build / Refresh index`.
3. Ask questions in the chat box.
4. Open `Sources used` to inspect the retrieved chunks.

Notes
-----

- If you later want LLM-generated answers, you can extend `build_answer()` with an API-backed summarizer.
- The current implementation works fully offline after installation and is suitable for Streamlit Cloud.

GitHub setup
------------

```bash
git init
git add .
git commit -m "Initial Smart FAQ Chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Streamlit Cloud
---------------

1. Push this repository to GitHub.
2. Open Streamlit Cloud and choose `New app`.
3. Select your repository, branch `main`, and set the main file path to `app.py`.
4. Deploy.

If you want private secrets later, add them in Streamlit Cloud as app secrets instead of committing them to GitHub.
