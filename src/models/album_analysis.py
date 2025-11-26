from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TopicInfo(BaseModel):
    main_topic_id: int
    topic_distribution: Dict[str, float]


class TopicDescription(BaseModel):
    top_words: List[str]
    label: str


class TopicsInfo(BaseModel):
    topics_per_track: Dict[str, TopicInfo]
    topic_descriptions: Dict[str, TopicDescription]
    album_topic_distribution: Dict[str, float]


class SentimentTrack(BaseModel):
    label: str
    score: float


class SentimentInfo(BaseModel):
    per_track: Dict[str, SentimentTrack]
    album_summary: Dict[str, Any]  # positive_ratio, etc.


class EmotionsTrack(BaseModel):
    sadness: float
    anger: float
    joy: float
    fear: float
    others: float


class EmotionsInfo(BaseModel):
    per_track: Dict[str, EmotionsTrack]
    album_emotional_profile: Dict[str, float]


class ContrastTrack(BaseModel):
    text_sentiment: str
    sadness: float
    valence: float
    danceability: float
    contrast_label: str


class ContrastInfo(BaseModel):
    per_track: Dict[str, ContrastTrack]
    album_contrast_summary: Dict[str, Any]


class ClusterInfo(BaseModel):
    cluster_id: int
    description: Optional[str] = None
    track_ids: List[str]


class SimilarityInfo(BaseModel):
    clusters: List[ClusterInfo]
    similarity_matrix_stats: Dict[str, float]
    most_atypical_track_id: Optional[str] = None


class KeywordsInfo(BaseModel):
    per_track: Dict[str, List[str]]
    album_keywords: List[str]


class AlbumSummary(BaseModel):
    main_themes: List[str]
    global_sentiment: str
    emotional_profile: Dict[str, float]
    most_atypical_track: Optional[Dict[str, str]] = None


class AlbumAnalysis(BaseModel):
    album_id: str
    generated_at: datetime
    analysis_version: int = 1
    models: Dict[str, str]
    
    topics_info: Optional[TopicsInfo] = None
    similarity_info: Optional[SimilarityInfo] = None
    keywords_info: Optional[KeywordsInfo] = None
    sentiment_info: Optional[SentimentInfo] = None
    emotions_info: Optional[EmotionsInfo] = None
    contrast_info: Optional[ContrastInfo] = None
    
    summary: Optional[AlbumSummary] = None
