# Deployment Architecture

```mermaid
flowchart TD
    browser["Browser"]
    vercel["React Website<br/>Vercel"]

    subgraph compose["Docker Compose"]
        streamlit["Streamlit Dashboard"]
        api["FastAPI Backend"]
        db["PostgreSQL Database"]
    end

    browser --> vercel
    browser --> streamlit
    browser --> api
    streamlit --> api
    api --> db
```
