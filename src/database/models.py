"""Database models for facial tracking system"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, LargeBinary, Boolean, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Person(Base):
    """Represents a tracked individual"""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    unique_id = Column(String(36), unique=True, nullable=False)  # UUID
    name = Column(String(200), nullable=True)

    # OSINT data
    detected_name = Column(String(200), nullable=True)  # Name found via OSINT
    profession = Column(String(200), nullable=True)
    nationality = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)

    # Tracking
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    total_sightings = Column(Integer, default=0)

    # Relationships
    faces = relationship("Face", back_populates="person")
    sightings = relationship("Sighting", back_populates="person")
    social_profiles = relationship("SocialProfile", back_populates="person")


class Face(Base):
    """Stores facial embeddings for a person"""
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # 512-dim vector as bytes
    confidence = Column(Float, default=0.0)
    source_image = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    person = relationship("Person", back_populates="faces")


class Sighting(Base):
    """Records when/where a person was seen"""
    __tablename__ = "sightings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)

    # Source info
    source_type = Column(String(50))  # 'image', 'video', 'webcam', etc.
    source_url = Column(String(1000), nullable=True)
    source_file = Column(String(500), nullable=True)

    # Location data (from EXIF)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_name = Column(String(200), nullable=True)

    # Temporal data
    captured_at = Column(DateTime, nullable=True)  # When photo was taken
    processed_at = Column(DateTime, default=datetime.utcnow)  # When we processed it

    # Raw EXIF
    exif_data = Column(Text, nullable=True)

    # Relationship
    person = relationship("Person", back_populates="sightings")


class SocialProfile(Base):
    """Social media profiles found via OSINT"""
    __tablename__ = "social_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)

    platform = Column(String(50))  # 'Instagram', 'Twitter', etc.
    username = Column(String(200), nullable=True)
    profile_url = Column(String(1000))
    profile_name = Column(String(200), nullable=True)
    confidence = Column(Float, default=0.0)
    found_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    person = relationship("Person", back_populates="social_profiles")


class ProcessedFile(Base):
    """Tracks processed files to avoid duplicates"""
    __tablename__ = "processed_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_hash = Column(String(64), unique=True, nullable=False)  # SHA256
    original_filename = Column(String(500))
    file_size = Column(Integer)

    # Processing info
    faces_detected = Column(Integer, default=0)
    persons_matched = Column(Integer, default=0)
    persons_new = Column(Integer, default=0)
    osint_completed = Column(Boolean, default=False)

    # Status
    processed_at = Column(DateTime, default=datetime.utcnow)
    moved_to = Column(String(500), nullable=True)  # Path in /processed/

    # Results
    status = Column(String(20), default='success')  # 'success', 'error', 'no_faces'
    error_message = Column(Text, nullable=True)


class ProcessingLog(Base):
    """Log of all processed images/sources"""
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_path = Column(String(1000), nullable=False)
    source_type = Column(String(50))
    file_hash = Column(String(64), nullable=True)  # SHA256

    faces_detected = Column(Integer, default=0)
    faces_matched = Column(Integer, default=0)
    faces_new = Column(Integer, default=0)

    processing_time = Column(Float)  # seconds
    status = Column(String(20))  # 'success', 'error', 'partial'
    error_message = Column(Text, nullable=True)

    processed_at = Column(DateTime, default=datetime.utcnow)