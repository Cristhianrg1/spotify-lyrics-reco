from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Optional

from src.clients.mongo_client import get_mongo
from src.services.track_service import TrackService
from src.models.album_analysis import (
    AlbumAnalysis,
    TopicsInfo,
    TopicInfo,
    TopicDescription,
    SentimentInfo,
    SentimentTrack,
    EmotionsInfo,
    EmotionsTrack,
    ContrastInfo,
    ContrastTrack,
    SimilarityInfo,
    ClusterInfo,
    KeywordsInfo,
    AlbumSummary,
)

# Optional imports for analysis
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    pass

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    if pipeline is None: return None
    # Multilingual sentiment model
    return pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

@lru_cache(maxsize=1)
def get_emotion_pipeline():
    if pipeline is None: return None
    # English emotion model (works decently for other languages usually, or we accept the limitation)
    # return_all_scores is deprecated, use top_k=None
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)


class AnalysisNotFound(Exception):
    pass


class AlbumAnalysisService:
    def __init__(self):
        self.mongo = get_mongo()
        self.track_service = TrackService(with_lyrics=True)

    async def get_or_create_analysis(
        self, album_id: str, create_if_missing: bool = False, force_recompute: bool = False
    ) -> AlbumAnalysis:
        existing = await self.mongo.album_analysis.find_one({"album_id": album_id})

        if existing and not force_recompute:
            return AlbumAnalysis(**existing)

        if not existing and not create_if_missing and not force_recompute:
            raise AnalysisNotFound(f"Analysis for album {album_id} not found")

        # Recompute or create
        return await self._run_pipeline(album_id)

    async def _run_pipeline(self, album_id: str) -> AlbumAnalysis:
        logger.info(f"Starting analysis pipeline for album_id={album_id}")
        
        # 1. Load data
        track_ids = await self.track_service.get_track_ids_for_album(album_id)
        logger.info(f"Found {len(track_ids)} tracks for album {album_id}")
        
        cursor = self.mongo.lyrics.find({"track_id": {"$in": track_ids}})
        lyrics_docs = await cursor.to_list(length=None)
        logger.info(f"Loaded {len(lyrics_docs)} lyrics documents")
        
        cursor_chunks = self.mongo.lyrics_chunks.find({"track_id": {"$in": track_ids}})
        chunks_docs = await cursor_chunks.to_list(length=None)
        logger.info(f"Loaded {len(chunks_docs)} embedding chunks")

        album_data = {
            "album_id": album_id,
            "tracks": {doc["track_id"]: doc for doc in lyrics_docs},
            "embeddings": {doc["track_id"]: doc for doc in chunks_docs},
        }

        # 2. Run sub-analyses
        logger.info("Running topic analysis...")
        topics_info = self._analyze_topics(album_data)
        
        logger.info("Running similarity analysis...")
        similarity_info = self._analyze_similarity(album_data)
        
        logger.info("Extracting keywords...")
        keywords_info = self._extract_keywords(album_data)
        
        logger.info("Running sentiment analysis...")
        sentiment_info = self._analyze_sentiment(album_data)
        
        logger.info("Running emotion analysis...")
        emotions_info = self._analyze_emotions(album_data)
        
        logger.info("Running contrast analysis...")
        contrast_info = self._analyze_contrast(album_data, sentiment_info)

        logger.info("Building summary...")
        summary = self._build_summary(
            album_data,
            topics_info,
            similarity_info,
            sentiment_info,
            emotions_info,
            contrast_info
        )

        analysis = AlbumAnalysis(
            album_id=album_id,
            generated_at=dt.datetime.utcnow(),
            models={
                "topic_model": "simple-kmeans-tfidf",
                "sentiment_model": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                "emotion_model": "j-hartmann/emotion-english-distilroberta-base"
            },
            topics_info=topics_info,
            similarity_info=similarity_info,
            keywords_info=keywords_info,
            sentiment_info=sentiment_info,
            emotions_info=emotions_info,
            contrast_info=contrast_info,
            summary=summary
        )

        # 3. Save
        await self.mongo.album_analysis.replace_one(
            {"album_id": album_id},
            analysis.model_dump(),
            upsert=True
        )
        
        logger.info(f"Analysis completed and saved for album {album_id}")

        return analysis

    def _get_stopwords(self) -> List[str]:
        # Basic English stopwords from sklearn are usually fine, but we want to add Spanish and fillers
        # This is a manual list for demo purposes. In prod, load from a file or library like NLTK/spacy
        spanish_stopwords = [
            "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como",
            "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
            "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
            "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada",
            "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras",
            "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra",
            "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están",
            "esté", "estés", "estemos", "estéis", "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría", "estarías",
            "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos",
            "estuvisteis", "estuvieron", "hubiera", "hubieras", "hubiéramos", "hubierais", "hubieran", "hubiese", "hubieses", "hubiésemos", "hubieseis",
            "hubiesen", "habiendo", "habido", "habida", "habidos", "habidas", "soy", "eres", "es", "somos", "sois", "son", "sea", "seas", "seamos", "seáis",
            "sean", "seré", "serás", "será", "seremos", "seréis", "serán", "sería", "serías", "seríamos", "seríais", "serían", "era", "eras", "éramos",
            "erais", "eran", "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron", "fuera", "fueras", "fuéramos", "fuerais", "fueran", "fuese", "fueses",
            "fuésemos", "fueseis", "fuesen", "sintiendo", "sentido", "sentida", "sentidos", "sentidas", "siente", "sentid", "tengo", "tienes", "tiene",
            "tenemos", "tenéis", "tienen", "tenga", "tengas", "tengamos", "tengáis", "tengan", "tendré", "tendrás", "tendrá", "tendremos", "tendréis",
            "tendrán", "tendría", "tendrías", "tendríamos", "tendríais", "tendrían", "tenía", "tenías", "teníamos", "teníais", "tenían", "tuve", "tuviste",
            "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera", "tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuviésemos",
            "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida", "tenidos", "tenidas", "tened"
        ]
        
        lyric_fillers = [
            "yeah", "oh", "baby", "ooh", "ah", "na", "la", "da", "di", "hey", "whoa", "uh", "huh", "let", "go", "get", "got", "wanna", "gonna", "ay", "tchu"
        ]
        
        return list(set(spanish_stopwords + lyric_fillers + list(TfidfVectorizer(stop_words="english").get_stop_words())))

    def _analyze_topics(self, album_data: Dict[str, Any]) -> TopicsInfo:
        tracks = album_data.get("tracks", {})
        if not tracks:
            return TopicsInfo(topics_per_track={}, topic_descriptions={}, album_topic_distribution={})

        # Prepare corpus
        track_ids = list(tracks.keys())
        corpus = [tracks[tid].get("lyrics_text", "") for tid in track_ids]
        
        n_tracks = len(track_ids)
        n_topics = min(3, n_tracks) if n_tracks > 0 else 1
        
        try:
            # Improved Vectorizer
            stopwords = self._get_stopwords()
            vectorizer = TfidfVectorizer(
                stop_words=stopwords,
                max_features=200, # Increased features
                ngram_range=(1, 2), # Unigrams and Bigrams
                min_df=2 if n_tracks > 5 else 1 # Ignore super rare words if we have enough tracks
            )
            X = vectorizer.fit_transform(corpus)
            
            # Store for similarity fallback
            self._last_tfidf_matrix = X
            self._last_track_ids = track_ids
            
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
            kmeans.fit(X)
            
            labels = kmeans.labels_
            feature_names = vectorizer.get_feature_names_out()
            
            # Topic descriptions
            topic_descriptions = {}
            for topic_idx in range(n_topics):
                center = kmeans.cluster_centers_[topic_idx]
                top_indices = center.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topic_descriptions[str(topic_idx)] = TopicDescription(
                    top_words=top_words,
                    label=f"Topic {topic_idx}"
                )
            
            # Per track info
            topics_per_track = {}
            topic_counts = {i: 0 for i in range(n_topics)}
            
            for idx, tid in enumerate(track_ids):
                topic_id = int(labels[idx])
                topics_per_track[tid] = TopicInfo(
                    main_topic_id=topic_id,
                    topic_distribution={str(topic_id): 1.0}
                )
                topic_counts[topic_id] += 1
                
            total = len(track_ids)
            album_dist = {str(k): v/total for k, v in topic_counts.items()}
            
            return TopicsInfo(
                topics_per_track=topics_per_track,
                topic_descriptions=topic_descriptions,
                album_topic_distribution=album_dist
            )
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return TopicsInfo(topics_per_track={}, topic_descriptions={}, album_topic_distribution={})

    def _analyze_similarity(self, album_data: Dict[str, Any]) -> SimilarityInfo:
        embeddings_map = album_data.get("embeddings", {})
        
        # Try to use embeddings first
        matrix = []
        valid_ids = []
        
        if embeddings_map:
            track_ids = list(embeddings_map.keys())
            for tid in track_ids:
                emb = embeddings_map[tid].get("embedding")
                if emb:
                    matrix.append(emb)
                    valid_ids.append(tid)
        
        X = None
        used_embeddings = False
        
        if matrix:
            X = np.array(matrix)
            used_embeddings = True
            logger.info("Using embeddings for similarity analysis")
        elif hasattr(self, "_last_tfidf_matrix") and self._last_tfidf_matrix is not None:
            # Fallback to TF-IDF
            X = self._last_tfidf_matrix
            valid_ids = self._last_track_ids
            logger.info("Using TF-IDF fallback for similarity analysis")
        else:
             return SimilarityInfo(clusters=[], similarity_matrix_stats={"avg_similarity": 0.0}, most_atypical_track_id=None)
             
        # Cosine similarity
        sim_matrix = cosine_similarity(X)
        
        # Stats
        avg_sim = float(np.mean(sim_matrix))
        
        # Clustering (simple)
        n_clusters = min(3, len(valid_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        
        clusters = []
        for cid in range(n_clusters):
            tids = [valid_ids[i] for i, label in enumerate(labels) if label == cid]
            clusters.append(ClusterInfo(cluster_id=cid, track_ids=tids, description=f"Cluster {cid}"))
            
        # Atypical track: lowest average similarity to others
        mean_sims = np.mean(sim_matrix, axis=1)
        min_idx = np.argmin(mean_sims)
        atypical_id = valid_ids[min_idx]
        
        return SimilarityInfo(
            clusters=clusters,
            similarity_matrix_stats={"avg_similarity": avg_sim},
            most_atypical_track_id=atypical_id
        )

    def _extract_keywords(self, album_data: Dict[str, Any]) -> KeywordsInfo:
        tracks = album_data.get("tracks", {})
        if not tracks:
            return KeywordsInfo(per_track={}, album_keywords=[])
            
        track_ids = list(tracks.keys())
        corpus = [tracks[tid].get("lyrics_text", "") for tid in track_ids]
        
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
            X = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            per_track = {}
            for idx, tid in enumerate(track_ids):
                # Get top words for this doc
                row = X[idx]
                # coo_matrix
                tuples = zip(row.col, row.data)
                sorted_items = sorted(tuples, key=lambda x: x[1], reverse=True)[:5]
                keywords = [feature_names[i] for i, score in sorted_items]
                per_track[tid] = keywords
                
            # Album keywords (global top)
            # Sum columns
            sum_scores = np.array(X.sum(axis=0)).flatten()
            top_indices = sum_scores.argsort()[-10:][::-1]
            album_keywords = [feature_names[i] for i in top_indices]
            
            return KeywordsInfo(per_track=per_track, album_keywords=album_keywords)
            
        except Exception:
            return KeywordsInfo(per_track={}, album_keywords=[])

    def _analyze_sentiment(self, album_data: Dict[str, Any]) -> SentimentInfo:
        tracks = album_data.get("tracks", {})
        per_track = {}
        
        sentiment_pipe = get_sentiment_pipeline()
        
        pos_count = 0
        neg_count = 0
        
        for tid, doc in tracks.items():
            text = doc.get("lyrics_text", "")
            # Truncate to avoid max length errors (BERT usually 512 tokens)
            # Approx 1500 chars is safe-ish
            text = text[:1500]
            
            if not text or not sentiment_pipe:
                label = "neutral"
                score = 0.5
            else:
                try:
                    # Returns list of dicts: [{'label': 'positive', 'score': 0.9}]
                    # Enable truncation to avoid "sequence length is longer than..." errors
                    result = sentiment_pipe(text, truncation=True, max_length=512)[0]
                    label = result["label"]
                    score = result["score"]
                except Exception as e:
                    logger.error(f"Sentiment error for {tid}: {e}")
                    label = "neutral"
                    score = 0.5

            # Normalize labels from model if needed
            # lxyuan model returns: "positive", "negative", "neutral"
            
            if label == "positive":
                pos_count += 1
            elif label == "negative":
                neg_count += 1
                
            per_track[tid] = SentimentTrack(label=label, score=score)
            
        total_tracks = len(tracks)
        summary = {
            "positive_ratio": pos_count / total_tracks if total_tracks else 0,
            "negative_ratio": neg_count / total_tracks if total_tracks else 0,
            "neutral_ratio": (total_tracks - pos_count - neg_count) / total_tracks if total_tracks else 0
        }
        
        return SentimentInfo(per_track=per_track, album_summary=summary)

    def _analyze_emotions(self, album_data: Dict[str, Any]) -> EmotionsInfo:
        tracks = album_data.get("tracks", {})
        per_track = {}
        
        emotion_pipe = get_emotion_pipeline()
        
        for tid, doc in tracks.items():
            text = doc.get("lyrics_text", "")
            text = text[:1500]
            
            scores = {
                "sadness": 0.0,
                "anger": 0.0,
                "joy": 0.0,
                "fear": 0.0,
                "others": 0.0
            }
            
            if text and emotion_pipe:
                try:
                    # Returns list of lists of dicts (return_all_scores=True equivalent)
                    # [[{'label': 'anger', 'score': 0.004}, ...]]
                    # Enable truncation to avoid length errors
                    results = emotion_pipe(text, truncation=True, max_length=512)[0]
                    
                    # Map model labels to our schema
                    # Model labels: anger, disgust, fear, joy, neutral, sadness, surprise
                    for r in results:
                        lbl = r["label"]
                        scr = r["score"]
                        
                        if lbl == "sadness":
                            scores["sadness"] = scr
                        elif lbl == "anger":
                            scores["anger"] = scr
                        elif lbl == "joy":
                            scores["joy"] = scr
                        elif lbl == "fear":
                            scores["fear"] = scr
                        else:
                            # disgust, neutral, surprise -> others
                            scores["others"] += scr
                            
                except Exception as e:
                    logger.error(f"Emotion error for {tid}: {e}")
                    scores["others"] = 1.0
            else:
                scores["others"] = 1.0
            
            per_track[tid] = EmotionsTrack(**scores)
            
        # Album profile: average
        if not per_track:
            return EmotionsInfo(per_track={}, album_emotional_profile={})
            
        avg_sadness = sum(t.sadness for t in per_track.values()) / len(per_track)
        avg_joy = sum(t.joy for t in per_track.values()) / len(per_track)
        avg_anger = sum(t.anger for t in per_track.values()) / len(per_track)
        avg_fear = sum(t.fear for t in per_track.values()) / len(per_track)
        avg_others = sum(t.others for t in per_track.values()) / len(per_track)
        
        return EmotionsInfo(
            per_track=per_track,
            album_emotional_profile={
                "sadness": avg_sadness,
                "joy": avg_joy,
                "anger": avg_anger,
                "fear": avg_fear,
                "others": avg_others
            }
        )

    def _analyze_contrast(self, album_data: Dict[str, Any], sentiment_info: SentimentInfo) -> ContrastInfo:
        # Need audio features. For now, we don't have them in album_data (only lyrics).
        # If we had them, we would compare.
        # Let's assume no audio features for now or mock them.
        return ContrastInfo(per_track={}, album_contrast_summary={"note": "Audio features not available yet"})

    def _build_summary(
        self,
        album_data: Dict[str, Any],
        topics: TopicsInfo,
        similarity: SimilarityInfo,
        sentiment: SentimentInfo,
        emotions: EmotionsInfo,
        contrast: ContrastInfo
    ) -> AlbumSummary:
        
        # Main themes
        themes = []
        if topics.topic_descriptions:
            themes = [d.label + ": " + ", ".join(d.top_words[:3]) for d in topics.topic_descriptions.values()]
            
        # Global sentiment
        s_summary = sentiment.album_summary
        if s_summary.get("positive_ratio", 0) > 0.5:
            glob_sent = "predominantly_positive"
        elif s_summary.get("negative_ratio", 0) > 0.5:
            glob_sent = "predominantly_negative"
        else:
            glob_sent = "mixed"
            
        # Atypical
        atypical = None
        if similarity.most_atypical_track_id:
            tid = similarity.most_atypical_track_id
            track = album_data["tracks"].get(tid)
            name = track.get("track_name", "Unknown") if track else "Unknown"
            atypical = {"track_id": tid, "track_name": name}
            
        return AlbumSummary(
            main_themes=themes,
            global_sentiment=glob_sent,
            emotional_profile=emotions.album_emotional_profile,
            most_atypical_track=atypical
        )
