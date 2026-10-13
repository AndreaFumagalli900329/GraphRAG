import sys
import json
import re
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
GRAPH_NAME = "football-graph"

class Neo4jLoader:

    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password), notifications_min_severity="OFF")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def close(self):
        self.driver.close()

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def split_text(self, text):
        char_size = CHUNK_SIZE * 5
        char_overlap = CHUNK_OVERLAP * 5

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_size,
            chunk_overlap=char_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
    
        return splitter.split_text(text)

    def insert_player(self, players):
        query = """
            UNWIND $players AS player
            MERGE (p:Player {id: player.id})
            SET
                p.name = player.name,
                p.age = player.age,
                p.team = player.team,
                p.nationality = player.nationality
            MERGE (pos:Position {
                name: player.position
            })
            MERGE (p)-[:PLAYS_AS]->(pos)
            """      
        with self.driver.session() as session:
            session.run(query, players=players)
    
    async def extract_traits(self, llm, player, chunk):
        stats = player.get("statistics", {})
        stats_formatted = "\n".join([
            f"- {key.replace('_', ' ').capitalize()}: {value}"
            for key, value in stats.items()
        ])

        prompt = f"""
            You are a professional football scout and match analysis expert.
            Your task is to extract the key characteristics (traits) of the player based on both the provided biographical/journalistic TEXT and his real performance STATISTICS.

            EXTRACTION RULES:
            1. Extract ONLY explicit qualitative attributes (e.g., "fast", "playmaker", "strong tackler", "prolific", "injury-prone").
            2. Use the STATISTICS to confirm or add qualitative traits. For example:
            - Many goals/assists -> "prolific goalscorer", "playmaker"
            - High passing accuracy -> "accurate passer"
            - Many tackles/interceptions -> "solid defensive player", "strong tackler"
            - Many successful dribbles -> "skilled dribbler"
            3. DO NOT include numbers or quantitative values in the traits (e.g., DO NOT write "scored 10 goals" or "90% accuracy").
            4. DO NOT invent traits that are not supported by either the text or the statistics.
            5. MANDATORY output.
                Respond EXCLUSIVELY with a JSON object containing a single list of strings, in this exact format:
                {{
                    "traits": ["fast", "strong tackler", "accurate passer"]
                }}

                No additional text.
                No explanation.
                No markdown.

            Player Data:
            - Name: {player.get('name')}
            - Role: {player.get('position')}
            - Age: {player.get('age')}

            Performance Statistics:
            {stats_formatted}

            Biographical Text/Article:
            {chunk}
            """
        try:
            json_llm = llm.bind(response_format={"typer": "json_object"})
            response = await json_llm.ainvoke(prompt)
            #print("RISPOSTA LLM:")
            #print(response.content)
            clean_content = response.content.replace("```json", "").replace("```", "").strip()

            data = json.loads(clean_content)
            if isinstance(data, list):
                return data
            
            if isinstance(data, dict):
                if "traits" in data:
                    return data["traits"]
                return [key for key, value in data.items() if str(value).lower() == "true"]
            return []
        except Exception as e:
            print("Errore parsing traits:", e)
            return []
        
    def add_trait(self, player_id, trait):
        query = """
            MATCH (p:Player {id:$id})
            UNWIND $traits AS trait
            MERGE (t:Trait {
                name: trait
            })
            MERGE (p)-[:HAS_TRAIT]->(t)
            """
        with self.driver.session() as session:
            session.run(query, id=player_id, traits=trait)

    def build_profile(self, player, traits):     
        stats = player.get("statistics", {})
        profile = f"""
            Player: {player["name"]}

            Basic Information:
            - Position: {player.get("position", "")}
            - Age: {player.get("age", "")}
            - Nationality: {player.get("nationality", "")}
            - Club: {player.get("team", "")}

            Playing Style:
            This player can be described with the following characteristics:
            {", ".join(traits) if traits else "No qualitative traits available."}

            Performance Statistics:
            - Appearances: {stats.get("appearances",0)}
            - Goals: {stats.get("goals",0)}
            - Assists: {stats.get("assists",0)}
            - Passes: {stats.get("passes",0)}
            - Tackles: {stats.get("tackles",0)}
            - Interceptions: {stats.get("interceptions",0)}
            - Dribbles: {stats.get("dribbles",0)}

            Overall scouting profile:
            This profile summarizes the player's tactical role, technical qualities,
            playing style and statistical production.
            """
        return profile.strip()
    
    def save_embedding(self, player_id, embedding, profile_text):
        query = """
            MATCH (p:Player {id:$id})
            SET p.embedding = $embedding,
                p.profile_text = $profile_text
            """
        with self.driver.session() as session:
            session.run(query, id=player_id, embedding=embedding, profile_text=profile_text)

    def create_fulltext_index(self):
        query = """
            CREATE FULLTEXT INDEX player_fulltext_idx IF NOT EXISTS
            FOR (p:Player)
            ON EACH [p.profile_text]
            """
        with self.driver.session() as session:
            session.run(query)

    async def enrich_players(self, llm, players):
        for player in players:
            articles = player.get("articles", [])
            text = "\n".join(a["text"] for a in articles if "text" in a)

            traits = set()

            if text.strip():
                chunks = self.split_text(text)
                for chunk in chunks:
                    #print(chunk)
                    extracted_traits = await self.extract_traits(llm, player, chunk)
                    #print(player["name"],"articoli:",len(articles),"lunghezza testo:",len(text))
                    if isinstance(extracted_traits, list):
                        traits.update(extracted_traits)
            
            traits_list = list(traits)
            #print(player["name"],traits_list)
            if traits_list:
                self.add_trait(player["id"], traits_list)

            profile = self.build_profile(player, traits_list)
            embedding = self.embedding_model.encode(profile).tolist()
            self.save_embedding(player["id"], embedding, profile)

    def create_vector_index(self):
        query = """
            CREATE VECTOR INDEX player_embeddings IF NOT EXISTS
            FOR (p:Player)
            ON (p.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`:384,
                    `vector.similarity_function`:'cosine'
                }
            }
            """
        with self.driver.session() as session:
            session.run(query)

    def create_similarity_edges(self, threshold=0.50):
        with self.driver.session() as session:
            try:
                session.run("CALL gds.graph.drop('player-similarity', false) YIELD graphName")
            except:
                pass

            projection_query = """
                CALL gds.graph.project(
                    'player-similarity',
                    {
                        Player: { label: 'Player' },
                        Trait: { label: 'Trait' },
                        Position: { label: 'Position' }
                    },
                    {
                        HAS_TRAIT: {
                            type: 'HAS_TRAIT',
                            orientation: 'NATURAL'
                        },
                        PLAYS_AS: {
                            type: 'PLAYS_AS',
                            orientation: 'NATURAL'
                        }
                    }
                )
                """
            session.run(projection_query)

            similarity_query = f"""
                CALL gds.nodeSimilarity.write(
                    'player-similarity',
                    {{
                        writeRelationshipType: 'SIMILAR_TO',
                        writeProperty: 'score',
                        similarityCutoff: {threshold},
                        topK: 5
                    }}
                )
                YIELD relationshipsWritten, nodesCompared
                """
            result = session.run(similarity_query).single()
            print(f"Relazioni SIMILAR_TO create: {result['relationshipsWritten']}")

    def find_graph_communities(self):
        with self.driver.session() as session:
            try:
                session.run("CALL gds.graph.drop('{GRAPH_NAME}', false) YIELD graphName RETURN graphName")
            except:
                pass

            projection_query = """
                    CALL gds.graph.project(
                        '{GRAPH_NAME}',
                        'Player',
                        {
                            SIMILAR_TO: {
                                orientation: 'UNDIRECTED',
                                properties: 'score'
                            }
                        }
                    )
                """
            session.run(projection_query)
            print("Grafo proiettato in memoria")

            leiden_query = """
                CALL gds.leiden.write(
                    '{GRAPH_NAME}',
                    {
                    relationshipWeightProperty: 'score',
                    writeProperty: 'community'
                    })
                YIELD communityCount, modularity
                """
            result = session.run(leiden_query).single()
            if result:
                print(f"Trovate {result['communityCount']} comunità distinte. Modularità: {result['modularity']:.4f}")
                return result['communityCount']
            else:
                print("Nessun risultato restituito dall'algoritmo Leiden.")
                return 0
        
    async def generate_communities_summaries(self, llm):
        query = """
            MATCH (p:Player)
            WHERE p.community IS NOT NULL
            RETURN p.community AS community_id,
                   collect({
                       name: p.name,
                       position: [(p)-[:PLAYS_AS]->(pos) | pos.name][0],
                       traits: [(p)-[:HAS_TRAIT]->(tr) | tr.name]
                   }) AS members
            ORDER BY community_id ASC
            """
        with self.driver.session() as session:
            records = session.run(query)
            communities = [record for record in records]

        summaries = {}

        for community in communities:
            c_id = community["community_id"]
            members = community["members"]
            members_text = ""
            all_traits = []
            for m in members:
                all_traits.extend(m["traits"])
                
            unique_traits = list(set([t for t in all_traits if t]))
            unique_traits_str = ", ".join(unique_traits) if unique_traits else "Nessuno"

            members_text = ""
            for m in members:
                traits_str = ", ".join(m['traits']) if m['traits'] else "Nessun tratto specifico"
                members_text += f"- {m['name']} (Ruolo: {m['position']}) | Tratti personali: [{traits_str}]\n"

            prompt = f"""
                You are a Sports Director and Chief Scout of a high-level football club.
                Analyze this community of football players that the algorithm has grouped based on the similarity of their qualitative and attitudinal traits.

                Community #{c_id}
                Global shared traits within the group: {unique_traits_str}

                List of members of the Community:
                {members_text}

                YOUR TASK:
                Write a descriptive summary of maximum 3-4 sentences that defines the IDENTIKIT of this community. 
                Explain the technical-tactical or character core that unites this group of players and what strategic utility they provide within a squad (e.g., "This group gathers the physical, ball-winning, and disruptive elements...", or "This community consists of creative and imaginative profiles...").
                
                MANDATORY RULES:
                - Respond in the Italian language.
                - Be discursive: DO NOT use JSON format, DO NOT enclose the response in curly or square brackets.
                - Provide the text of the summary DIRECTLY, without introductions, preambles, or greetings.
                """
            
            try:
                response = await llm.ainvoke(prompt)
                summary_content = response.content.strip()
                if summary_content.startswith("{"):
                    match = re.search(r'["\'](.*?)["\']\s*:', summary_content)
                    if match:
                        summary_content = match.group(1)
                    else:
                        summary_content = summary_content.replace("{", "").replace("}", "").replace('"', '').strip()
                
                summaries[c_id] = summary_content
                #print(f"\n--- SOMMARIO COMUNITÀ #{c_id} ---")
                #print(summary_content)
                
                self.save_community_summary_to_db(c_id, summary_content)
            except Exception as e:
                print(f"Errore nella generazione del sommario per la comunità {c_id}: {e}")

        return summaries
    
    def save_community_summary_to_db(self, community_id, summary):
        query = """
            MERGE (c:Community {id: $community_id})
            SET c.summary = $summary,
                c.updatedAt = datetime()
            WITH c
            MATCH (p:Player)
            WHERE p.community = $community_id
            MERGE (p)-[:BELONGS_TO_COMMUNITY]->(c)
            """
        with self.driver.session() as session:
            try:
                session.run(
                    query, 
                    community_id=int(community_id), 
                    summary=summary
                )
            except Exception as e:
                print(f"Errore durante il salvataggio della comunità {community_id} nel DB: {e}")

    async def build_graph(self, players, llm):
        print("Inserimento giocatori e ruoli...")
        self.insert_player(players)

        print("Inserimento caratteristiche giocatori tramite LLM e generazione embedding...")
        await self.enrich_players(llm, players)

        print("Creazione indice vettoriale...")
        self.create_vector_index()

        print("Creazione indice full-text...")
        self.create_fulltext_index()

        print("Creazione relazioni di similarità...")
        self.create_similarity_edges()

        print("Individuazione delle comunità all'interno del grafo...")
        self.find_graph_communities()

        print("Generazione sommari delle comunità tramite LLM...")
        await self.generate_communities_summaries(llm)

        print("Grafo completato.")

    async def rewrite_query(self, query, llm):
        system_prompt = """
            You are an AI assistant specialized in data ingestion and retrieval for a football/soccer GraphRAG system.
            Your task is to analyze the user's request and break it down into a structured JSON object.
            
            You must map and normalize the information according to these STRICT RULES:
            1. 'search_terms': A string in Italian containing the key concepts expanded with synonyms useful for a Full-Text search (e.g., "regista" becomes "regista metodista centrocampista creatività").
            2. 'explicit_roles': A list of roles if mentioned, but you must categorize them RIGIDLY using ONLY these 4 allowed values: ["goalkeeper", "defender", "midfielder", "attacker"]. Do not invent other roles.
            3. 'expected_traits': A list of technical/tactical characteristics translated into ENGLISH (e.g., "forte nei contrasti" -> "strong tackler", "passaggi precisi" -> "accurate passer").
            4. 'english_query_for_embedding': An optimized summary in ENGLISH of the requested player's profile. This will be used to generate the vector embedding with the bge-en model.

            Respond EXCLUSIVELY with a valid JSON object, without markdown formatting (no ```json), and without explanations.
            
            Example of required output:
            {
            "search_terms": "terzino sinistro spinta corsa cross veloce",
            "explicit_roles": ["defender"],
            "expected_traits": ["fast pacing", "good crossing"],
            "english_query_for_embedding": "A fast defender capable of accurate crossing and high pace"
            }
            """
        try:
            json_llm = llm.bind(response_format={"type": "json_object"})
            response = await json_llm.ainvoke([
                ("system", system_prompt),
                ("user", f"Rielabora questa richiesta: '{query}'")
            ])
            clean_content = response.content.strip().replace("```json", "").replace("```", "")
            import json
            structured_query = json.loads(clean_content)
            
            if "explicit_roles" in structured_query:
                structured_query["explicit_roles"] = [r.lower() for r in structured_query["explicit_roles"]]
            
            return structured_query
        except Exception as e:
            print(f"Errore durante il rewriting della query: {e}")
            return {
                "search_terms": query,
                "explicit_roles": [],
                "expected_traits": [],
                "english_query_for_embedding": query
            }
        
    def _retrieve_hybrid_context(self, structured_query, k=5):
        english_query = structured_query.get("english_query_for_embedding", "")
        search_terms = structured_query.get("search_terms", "")
        explicit_roles = [r.lower() for r in structured_query.get("explicit_roles", [])]
        expected_traits = [t.lower() for t in structured_query.get("expected_traits", [])]
        query_vector = self.embedding_model.encode(english_query).tolist()
        candidates = {}

        with self.driver.session() as session:
            q_vector = """
                CALL db.index.vector.queryNodes("player_embeddings", $limit, $vector)
                YIELD node, score
                RETURN node.name AS name, node, toFloat(score) AS score
                """
            try:
                res_v = session.run(q_vector, vector=query_vector, limit=k*2)
                for r in res_v:
                    name = r["name"]
                    candidates[name] = {"node": r["node"], "v_score": r["score"], "t_score": 0.0}
            except Exception as e:
                print(f"[DEBUG] Errore o indice vettoriale vuoto: {e}")

            q_text = """
            CALL db.index.fulltext.queryNodes("player_fulltext_idx", $search_terms)
            YIELD node, score
            RETURN node.name AS name, node, toFloat(score) AS score
            LIMIT $limit
            """
            try:
                res_t = session.run(q_text, search_terms=search_terms, limit=k*2)
                for r in res_t:
                    name = r["name"]
                    if name in candidates:
                        candidates[name]["t_score"] = r["score"]
                    else:
                        candidates[name] = {"node": r["node"], "v_score": 0.0, "t_score": r["score"]}
            except Exception as e:
                print(f"[DEBUG] Errore o indice Full-Text vuoto: {e}")

            if not candidates:
                print("[DEBUG] Nessun giocatore trovato da entrambi gli indici.")
                return "Nessun calciatore corrispondente trovato nel database."

            max_v = max([c["v_score"] for c in candidates.values()]) or 1.0
            max_t = max([c["t_score"] for c in candidates.values()]) or 1.0

            scored_players = []

            for name, data in candidates.items():
                norm_v = data["v_score"] / max_v
                norm_t = data["t_score"] / max_t
                
                q_details = """
                MATCH (p:Player) WHERE elementId(p) = $node_element_id
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                OPTIONAL MATCH (p)-[:HAS_TRAIT]->(t:Trait)
                OPTIONAL MATCH (p)-[:BELONGS_TO_COMMUNITY]->(c:Community)
                OPTIONAL MATCH (p)-[s:SIMILAR_TO]-(sim:Player)
                RETURN collect(DISTINCT pos.name) AS positions,
                       collect(DISTINCT t.name) AS traits,
                       c.id AS c_id,
                       c.summary AS c_summary,
                       collect(DISTINCT sim.name)[..2] AS top_similar
                """
                res_details = session.run(q_details, node_element_id=data["node"].element_id)
                details = res_details.single()

                positions = details["positions"] if details else []
                traits = details["traits"] if details else []
                c_id = details["c_id"] if details and details["c_id"] is not None else "N/D"
                c_summary = details["c_summary"] if details and details["c_summary"] else "Nessun cluster tattico."
                top_similar = details["top_similar"] if details else []

                role_bonus = sum([0.5 for r in positions if r.lower() in explicit_roles])
                trait_bonus = sum([0.5 for t in traits if t.lower() in expected_traits])

                final_score = norm_v + norm_t + role_bonus + trait_bonus
                
                scored_players.append({
                    "name": name,
                    "final_score": final_score,
                    "positions": positions,
                    "traits": traits,
                    "c_id": c_id,
                    "c_summary": c_summary,
                    "top_similar": top_similar
                })

            scored_players = sorted(scored_players, key=lambda x: x["final_score"], reverse=True)[:k]

            context_chunks = []
            for p in scored_players:
                chunk = f"""
                --- PROFILO CALCIATORE ESTRATTO: {p['name']} (Match Score: {p['final_score']:.2f}) ---
                • Ruoli effettivi nel grafo: {', '.join(p['positions']) if p['positions'] else 'Non specificato'}
                • Tratti accertati: {', '.join(p['traits']) if p['traits'] else 'Nessuno'}
                • Affinità con (Node Similarity): {', '.join(p['top_similar']) if p['top_similar'] else 'Nessuno'}
                • Inquadramento nel Gruppo Tattico (Comunità #{p['c_id']}): 
                  {p['c_summary']}
                """
                context_chunks.append(chunk.strip())

            return "\n\n".join(context_chunks)
        
    async def generate_final_response(self, user_query, context, llm):
        system_prompt = """
            You are an expert Sports Director and Match Analyst. Your task is to answer the user's request
            using EXCLUSIVELY the verified information extracted from the data graph (Knowledge Graph) provided in the Context.
            
            Rules for the response:
            1. Be professional, analytical, and precise in your football evaluations.
            2. For each player mentioned in the context, explain why they are relevant to the request, citing their roles, traits, and similarities.
            3. If Tactical Community/Cluster information is present in the context, use it to enrich the response, without citing them directly.
            4. Do not hallucinate or invent data not present in the context. If information on a player is partial, limit yourself to what is written.
            5. Provide the response in the same language as the user's request and exclusively in text format.
            
            GRAPH CONTEXT:
            {context}
            """
        try:
            response = await llm.ainvoke([
                ("system", system_prompt.format(context=context)),
                ("user", f"Richiesta dell'utente: '{user_query}'")
            ])
            return response.content
        except Exception as e:
            print(f"Errore durante la generazione finale: {e}")
            return f"Non è stato possibile generare la risposta a causa di un errore: {e}"
    
    async def retrieve_players(self, query, llm):
        print("Rielaborazione della query utente...")
        structured_query = await self.rewrite_query(query, llm)

        print("Estrazione del contesto dei giocatori...")
        context = self._retrieve_hybrid_context(structured_query)
        #print(f"Contesto generato: \n{context}")
        
        print("Generazione del prompt di risposta...")
        response = await self.generate_final_response(query, context, llm)
        return response