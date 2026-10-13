import streamlit as st
import asyncio
from datetime import datetime

class FootballGraphUI:
    def __init__(self, neo4j_loader, llm, api_client, article_retriever, pipeline_func, retrieve_from_team_api_func, retrieve_from_league_api_func):
        self.neo4j_loader = neo4j_loader
        self.llm = llm
        self.api = api_client
        self.article_retriever = article_retriever
        self.pipeline_func = pipeline_func
        self.retrieve_from_team_api_func = retrieve_from_team_api_func
        self.retrieve_from_league_api_func = retrieve_from_league_api_func

    def _get_penultimate_season(self):
        """Calcola automaticamente l'anno della penultima stagione rispetto all'anno corrente."""
        current_year = datetime.now().year
        return current_year - 2  # Se siamo nel 2026, restituisce 2024

    def render(self):
        st.set_page_config(page_title="AI Football Scout & GraphRAG", page_icon="⚽", layout="wide")
        
        st.title("⚽ AI Football Scout - GraphRAG Control Panel")
        st.subheader("Gestione dati, Ingestion e Chatbot di Scouting")
        
        # Calcolo automatico della stagione
        target_season = self._get_penultimate_season()

        # --- SIDEBAR DI CONTROLLO ---
        st.sidebar.header("🛠️ Operazioni sul Grafo")
        st.sidebar.info(f"Stagione target calcolata: **{target_season}**")

        available_teams = self.api.get_available_team_names()
        selected_team = st.sidebar.selectbox(
            "Seleziona la squadra da aggiornare",
            options=available_teams,
            index=available_teams.index("Lazio") if "Lazio" in available_teams else 0,
            disabled=not available_teams,
            help="Le squadre disponibili vengono lette dal file team_id.json"
        )

        # 1. Pulsante Aggiornamento API per singola squadra
        if st.sidebar.button("📥 Aggiorna JSON da API", use_container_width=True):
            with st.spinner(f"Recupero dati stagione {target_season} in corso..."):
                try:
                    team_id = self.api.get_team_id_by_name(selected_team)
                    if team_id is None:
                        st.sidebar.error("❌ La squadra selezionata non ha un team_id valido nel mapping.")
                    else:
                        players = self.retrieve_from_team_api_func(self.api, self.article_retriever, target_season, team_id=team_id)
                        st.sidebar.success(f"✅ File JSON aggiornato per {selected_team} con {len(players)} giocatori!")
                except Exception as e:
                    st.sidebar.error(f"❌ Errore durante il recupero API: {e}")

        # 2. Pulsante Aggiornamento completo della lega
        if st.sidebar.button("🌍 Aggiorna tutto il file JSON dalla Serie A", use_container_width=True):
            with st.spinner(f"Recupero di tutti i giocatori della Serie A stagione {target_season} in corso..."):
                try:
                    players = self.retrieve_from_league_api_func(self.api, self.article_retriever, target_season, league_id=135)
                    if not players:
                        st.sidebar.error("❌ Nessun giocatore recuperato dalla API.")
                    else:
                        st.sidebar.success(f"✅ File JSON aggiornato con {len(players)} giocatori della Serie A!")
                except Exception as e:
                    st.sidebar.error(f"❌ Errore durante l'aggiornamento completo del JSON: {e}")

        st.sidebar.markdown("---")

        # 3. Pulsante Generazione / Rigenerazione Grafo
        if st.sidebar.button("🔄 Rigenera Grafo Neo4j", use_container_width=True):
            with st.spinner("Svuotamento grafo e ricostruzione semantica..."):
                try:
                    # Esecuzione della pipeline asincrona usando il loop corrente di Streamlit
                    asyncio.run(self.pipeline_func(self.neo4j_loader, self.llm))
                    st.sidebar.success("✅ Grafo Neo4j rigenerato e indicizzato!")
                except Exception as e:
                    st.sidebar.error(f"❌ Errore durante la pipeline: {e}")

        # --- AREA CHATBOT PRINCIPALE ---
        st.markdown("### 🔍 Interroga lo Scout AI")
        
        # Inizializzazione della cronologia dei messaggi nella sessione di Streamlit
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Mostra i messaggi precedenti
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input dell'utente
        if user_query := st.chat_input("Chiedi un identikit (es: 'Cerco difensori centrali fisici e bravi ad intercettare')"):
            # Mostra il messaggio dell'utente
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Genera la risposta
            with st.chat_message("assistant"):
                with st.spinner("Analisi del grafo e generazione del report in corso..."):
                    try:
                        # Invocazione asincrona del metodo del loader
                        answer = asyncio.run(self.neo4j_loader.retrieve_players(user_query, self.llm))
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"Si è verificato un errore durante il retrieval: {e}"
                        st.error(error_msg)