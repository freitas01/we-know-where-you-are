"""
We Know Where You Are - Database Models
SQLAlchemy models for storing face data and tracking information
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Person(Base):
    """Unique person identified by facial recognition"""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unique_id = Column(String(36), unique=True, nullable=False)  # UUID
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_sightings = Column(Integer, default=1)

    # Optional identified info (from OSINT)
    name = Column(String(255), nullable=True)
    possible_names = Column(Text, nullable=True)  # JSON list

    # Relationships
    faces = relationship("Face", back_populates="person")
    sightings = relationship("Sighting", back_populates="person")
    social_profiles = relationship("SocialProfile", back_populates="person")


class Face(Base):
    """Face embedding/encoding for a person"""
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Numpy array as bytes
    confidence = Column(Float, default=1.0)
    source_image = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    person = relationship("Person", back_populates="faces")


class Sighting(Base):
    """Record of where/when a person was seen"""
    __tablename__ = "sightings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)

    # Source info
    source_type = Column(String(50))  # 'image', 'video', 'stream', 'social_media'
    source_url = Column(String(1000), nullable=True)
    source_file = Column(String(500), nullable=True)

    # Location data
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_name = Column(String(255), nullable=True)

    # Time data
    captured_at = Column(DateTime, nullable=True)  # When photo/video was taken
    processed_at = Column(DateTime, default=datetime.utcnow)

    # Image metadata
    exif_data = Column(Text, nullable=True)  # JSON

    # Relationship
    person = relationship("Person", back_populates="sightings")


class SocialProfile(Base):
    """Social media profiles linked to a person"""
    __tablename__ = "social_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)

    platform = Column(String(50))  # 'instagram', 'facebook', 'tiktok', etc
    username = Column(String(255), nullable=True)
    profile_url = Column(String(500), nullable=True)
    profile_name = Column(String(255), nullable=True)
    confidence = Column(Float, default=0.0)  # How sure we are this is the same person

    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_checked = Column(DateTime, default=datetime.utcnow)

    # Relationship
    person = relationship("Person", back_populates="social_profiles")


class ProcessingLog(Base):
    """Log of all processed images/sources"""
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_path = Column(String(1000), nullable=False)
    source_type = Column(String(50))
    faces_detected = Column(Integer, default=0)
    faces_matched = Column(Integer, default=0)
    faces_new = Column(Integer, default=0)
    processing_time = Column(Float)  # seconds
    status = Column(String(20))  # 'success', 'error', 'partial'
    error_message = Column(Text, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow)