import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from src.database.repository import Repository
from src.database.models import Person  # Importação necessária para o fix

# Page Config
st.set_page_config(
    page_title="We Know Where You Are - Dashboard",
    page_icon="👁️",
    layout="wide"
)

# Header
st.title("👁️ We Know Where You Are")
st.markdown("**Facial Tracking & OSINT System | Sistema de Rastreamento Facial e OSINT**")
st.markdown("---")


def load_data():
    """Carrega dados do banco / Loads data from database"""
    repo = Repository()

    # Busca estatísticas
    stats = repo.get_stats()

    # CORREÇÃO AQUI: Abrimos uma sessão manual para usar 'joinedload'
    session = repo.get_session()
    try:
        # joinedload obriga o banco a trazer os 'sightings' junto com a 'Person'
        # Isso evita o erro de "Session closed"
        persons = session.query(Person).options(joinedload(Person.sightings)).all()

        data = []
        for p in persons:
            # Pega o último avistamento para exibir
            # Agora p.sightings já está carregado na memória!
            last_sighting = p.sightings[-1] if p.sightings else None
            source = last_sighting.source_file if last_sighting else "Unknown"

            data.append({
                "ID": p.id,
                "UUID": p.unique_id,
                "Name/Tag": p.name,
                "First Seen": p.first_seen,
                "Last Seen": p.last_seen,
                "Sightings": p.total_sightings,
                "Last Source": source
            })

        return stats, pd.DataFrame(data)
    finally:
        session.close()


# Sidebar - Statistics
st.sidebar.header("System Status | Status do Sistema")
try:
    stats, df = load_data()

    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Persons/Pessoas", stats['total_persons'])
    col2.metric("Faces", stats['total_faces'])
    col3.metric("Sightings/Vistos", stats['total_sightings'])

    st.sidebar.success("Database Connected | Banco Conectado")

except Exception as e:
    st.sidebar.error(f"Database Error: {e}")
    st.stop()

# Main Content
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📋 Tracking Registry | Registro de Rastreamento")
    if not df.empty:
        # Exibe a tabela formatada
        st.dataframe(
            df,
            column_config={
                "First Seen": st.column_config.DatetimeColumn(format="D/M/Y, h:m:s"),
                "Last Seen": st.column_config.DatetimeColumn(format="D/M/Y, h:m:s"),
            },
            use_container_width=True
        )
    else:
        st.info("Database is empty. Run ingest_faces.py to add targets. | Banco vazio. Rode ingest_faces.py.")

with col_right:
    st.subheader("🔍 System Logs | Logs do Sistema")
    st.code("""
    [INFO] System initialized...
    [INFO] Connected to SQLite...
    [INFO] RetinaFace model loaded...
    [INFO] ArcFace model loaded...
    """, language="bash")

    st.warning("⚠️ Privacy Alert: This system processes biometric data.")

# Footer
st.markdown("---")
st.caption(
    "Bachelor's Thesis Project - Systems Analysis and Development | Projeto de TCC - Análise e Desenvolvimento de Sistemas")