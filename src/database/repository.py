"""Database operations and repository pattern implementation"""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from src.config import DATABASE_URL
from src.database.models import Base, Person, Face, Sighting, SocialProfile, ProcessedFile, ProcessingLog


class Repository:
    """Repository for all database operations"""

    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    # ==================== PERSON OPERATIONS ====================

    def create_person(self, name: Optional[str] = None) -> Person:
        """Create a new person with unique UUID"""
        with self.get_session() as session:
            person = Person(
                unique_id=str(uuid.uuid4()),
                name=name,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                total_sightings=0
            )
            session.add(person)
            session.commit()
            session.refresh(person)
            return person

    def get_person_by_id(self, person_id: int) -> Optional[Person]:
        """Get person by database ID"""
        with self.get_session() as session:
            return session.query(Person).filter(Person.id == person_id).first()

    def get_person_by_uuid(self, unique_id: str) -> Optional[Person]:
        """Get person by UUID"""
        with self.get_session() as session:
            return session.query(Person).filter(Person.unique_id == unique_id).first()

    def get_all_persons(self) -> List[Person]:
        """Get all persons in database"""
        with self.get_session() as session:
            return session.query(Person).all()

    def update_person_sighting(self, person_id: int) -> None:
        """Update last_seen and increment sighting count"""
        with self.get_session() as session:
            person = session.query(Person).filter(Person.id == person_id).first()
            if person:
                person.last_seen = datetime.utcnow()
                person.total_sightings += 1
                session.commit()

    def update_person_osint(self, person_id: int, detected_name: str = None,
                            profession: str = None, nationality: str = None,
                            description: str = None) -> None:
        """Update person with OSINT data"""
        with self.get_session() as session:
            person = session.query(Person).filter(Person.id == person_id).first()
            if person:
                if detected_name:
                    person.detected_name = detected_name
                if profession:
                    person.profession = profession
                if nationality:
                    person.nationality = nationality
                if description:
                    person.description = description
                session.commit()

    # ==================== FACE OPERATIONS ====================

    def add_face_to_person(self, person_id: int, embedding: bytes,
                           confidence: float, source_image: str) -> Face:
        """Add a face embedding to a person"""
        with self.get_session() as session:
            face = Face(
                person_id=person_id,
                embedding=embedding,
                confidence=confidence,
                source_image=source_image
            )
            session.add(face)
            session.commit()
            session.refresh(face)
            return face

    def get_all_face_embeddings(self) -> List[tuple]:
        """Get all face embeddings with person IDs for matching"""
        with self.get_session() as session:
            faces = session.query(Face.person_id, Face.embedding).all()
            return [(f.person_id, f.embedding) for f in faces]

    # ==================== SIGHTING OPERATIONS ====================

    def add_sighting(self, person_id: int, source_type: str,
                     source_url: Optional[str] = None,
                     source_file: Optional[str] = None,
                     latitude: Optional[float] = None,
                     longitude: Optional[float] = None,
                     location_name: Optional[str] = None,
                     captured_at: Optional[datetime] = None,
                     exif_data: Optional[str] = None) -> Sighting:
        """Record a sighting of a person"""
        with self.get_session() as session:
            sighting = Sighting(
                person_id=person_id,
                source_type=source_type,
                source_url=source_url,
                source_file=source_file,
                latitude=latitude,
                longitude=longitude,
                location_name=location_name,
                captured_at=captured_at or datetime.utcnow(),
                exif_data=exif_data
            )
            session.add(sighting)
            session.commit()

            # Update person's sighting count
            self.update_person_sighting(person_id)

            session.refresh(sighting)
            return sighting

    def get_person_sightings(self, person_id: int) -> List[Sighting]:
        """Get all sightings for a person"""
        with self.get_session() as session:
            return session.query(Sighting).filter(
                Sighting.person_id == person_id
            ).order_by(Sighting.captured_at.desc()).all()

    # ==================== SOCIAL PROFILE OPERATIONS ====================

    def add_social_profile(self, person_id: int, platform: str,
                           profile_url: str, username: str = None,
                           profile_name: Optional[str] = None,
                           confidence: float = 0.0) -> SocialProfile:
        """Add a social media profile to a person"""
        with self.get_session() as session:
            # Check if profile already exists
            existing = session.query(SocialProfile).filter(
                SocialProfile.person_id == person_id,
                SocialProfile.platform == platform,
                SocialProfile.profile_url == profile_url
            ).first()

            if existing:
                existing.confidence = max(existing.confidence, confidence)
                session.commit()
                return existing

            profile = SocialProfile(
                person_id=person_id,
                platform=platform,
                username=username,
                profile_url=profile_url,
                profile_name=profile_name,
                confidence=confidence
            )
            session.add(profile)
            session.commit()
            session.refresh(profile)
            return profile

    # ==================== PROCESSED FILE OPERATIONS ====================

    def is_file_processed(self, file_hash: str) -> bool:
        """Check if file was already processed by SHA256 hash"""
        with self.get_session() as session:
            exists = session.query(ProcessedFile).filter(
                ProcessedFile.file_hash == file_hash
            ).first()
            return exists is not None

    def add_processed_file(self, file_hash: str, original_filename: str,
                           file_size: int, faces_detected: int = 0,
                           persons_matched: int = 0, persons_new: int = 0,
                           osint_completed: bool = False,
                           moved_to: str = None, status: str = 'success',
                           error_message: str = None) -> ProcessedFile:
        """Record a processed file"""
        with self.get_session() as session:
            processed = ProcessedFile(
                file_hash=file_hash,
                original_filename=original_filename,
                file_size=file_size,
                faces_detected=faces_detected,
                persons_matched=persons_matched,
                persons_new=persons_new,
                osint_completed=osint_completed,
                moved_to=moved_to,
                status=status,
                error_message=error_message
            )
            session.add(processed)
            session.commit()
            session.refresh(processed)
            return processed

    def get_processed_file(self, file_hash: str) -> Optional[ProcessedFile]:
        """Get processed file record by hash"""
        with self.get_session() as session:
            return session.query(ProcessedFile).filter(
                ProcessedFile.file_hash == file_hash
            ).first()

    # ==================== PROCESSING LOG OPERATIONS ====================

    def log_processing(self, source_path: str, source_type: str,
                       file_hash: str = None, faces_detected: int = 0,
                       faces_matched: int = 0, faces_new: int = 0,
                       processing_time: float = 0.0, status: str = 'success',
                       error_message: Optional[str] = None) -> ProcessingLog:
        """Log a processing operation"""
        with self.get_session() as session:
            log = ProcessingLog(
                source_path=source_path,
                source_type=source_type,
                file_hash=file_hash,
                faces_detected=faces_detected,
                faces_matched=faces_matched,
                faces_new=faces_new,
                processing_time=processing_time,
                status=status,
                error_message=error_message
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log

    # ==================== STATISTICS ====================

    def get_stats(self) -> dict:
        """Get database statistics"""
        with self.get_session() as session:
            return {
                'total_persons': session.query(func.count(Person.id)).scalar(),
                'total_faces': session.query(func.count(Face.id)).scalar(),
                'total_sightings': session.query(func.count(Sighting.id)).scalar(),
                'total_social_profiles': session.query(func.count(SocialProfile.id)).scalar(),
                'total_processed': session.query(func.count(ProcessedFile.id)).scalar()
            }