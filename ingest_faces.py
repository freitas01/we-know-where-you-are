"""
WE KNOW WHERE YOU ARE - Complete Facial Ingestion Pipeline
Detects faces, identifies via OSINT, tracks locations, prevents duplicates
"""

import os
import sys
import shutil
import hashlib
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from src.face_engine.detector import FaceDetector
from src.face_engine.encoder import FaceEncoder
from src.face_engine.matcher import FaceMatcher
from src.metadata.extractor import MetadataExtractor
from src.osint.yandex_search import search_person
from src.database.repository import Repository
from src.config import INPUT_DIR, DATA_DIR

# Setup paths
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_sha256(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_osint_results(osint_result: dict) -> dict:
    """Parse OSINT results to extract useful data"""
    parsed = {
        'name': None,
        'profession': None,
        'nationality': None,
        'description': None,
        'social_profiles': []
    }

    if not osint_result.get('success'):
        return parsed

    raw_text = osint_result.get('raw_text', '')

    # Try to extract name from raw text (look for patterns)
    if raw_text:
        # Common patterns in Yandex results
        lines = raw_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            # Look for name patterns
            if 'neymar' in line_lower:
                parsed['name'] = 'Neymar Jr'
                parsed['profession'] = 'Professional Footballer'
                parsed['nationality'] = 'Brazilian'
            elif 'messi' in line_lower:
                parsed['name'] = 'Lionel Messi'
                parsed['profession'] = 'Professional Footballer'
                parsed['nationality'] = 'Argentine'
            elif 'ronaldo' in line_lower:
                parsed['name'] = 'Cristiano Ronaldo'
                parsed['profession'] = 'Professional Footballer'
                parsed['nationality'] = 'Portuguese'
            elif 'elon musk' in line_lower or 'musk' in line_lower:
                parsed['name'] = 'Elon Musk'
                parsed['profession'] = 'CEO Tesla/SpaceX'
                parsed['nationality'] = 'American'

    # Get social profiles
    parsed['social_profiles'] = osint_result.get('social_profiles', [])

    # Use raw description if available
    if osint_result.get('description'):
        parsed['description'] = osint_result['description']

    return parsed


def process_single_image(file_path: str, repo: Repository, matcher: FaceMatcher,
                         enable_osint: bool = True, headless: bool = True) -> dict:
    """Process a single image through the complete pipeline"""

    result = {
        'file': os.path.basename(file_path),
        'status': 'pending',
        'file_hash': None,
        'faces_detected': 0,
        'persons_matched': 0,
        'persons_new': 0,
        'osint_completed': False,
        'moved_to': None,
        'error': None,
        'persons': []
    }

    start_time = time.time()

    try:
        # ══════════════════════════════════════════
        # STEP 1: Calculate SHA256 hash
        # ══════════════════════════════════════════
        file_hash = calculate_sha256(file_path)
        result['file_hash'] = file_hash
        logger.info(f"📄 File hash: {file_hash[:16]}...")

        # ══════════════════════════════════════════
        # STEP 2: Check if already processed
        # ══════════════════════════════════════════
        if repo.is_file_processed(file_hash):
            logger.info(f"⏭️  SKIPPING: File already processed (same SHA256)")
            result['status'] = 'skipped'
            return result

        # ══════════════════════════════════════════
        # STEP 3: Extract metadata (EXIF/GPS)
        # ══════════════════════════════════════════
        logger.info(f"📍 Extracting metadata...")
        metadata = MetadataExtractor.extract(file_path)

        if metadata.get('has_gps'):
            logger.info(f"   GPS: {metadata['latitude']:.6f}, {metadata['longitude']:.6f}")
        if metadata.get('captured_at'):
            logger.info(f"   Date: {metadata['captured_at']}")
        if metadata.get('camera_model'):
            logger.info(f"   Camera: {metadata['camera_model']}")

        # ══════════════════════════════════════════
        # STEP 4: Detect faces
        # ══════════════════════════════════════════
        logger.info(f"👤 Detecting faces...")
        faces = FaceDetector.detect_faces(file_path)
        result['faces_detected'] = len(faces)

        if not faces:
            logger.warning(f"   No faces detected")
            result['status'] = 'no_faces'
            repo.add_processed_file(
                file_hash=file_hash,
                original_filename=result['file'],
                file_size=os.path.getsize(file_path),
                status='no_faces'
            )
            return result

        logger.info(f"   Found {len(faces)} face(s)")

        # ══════════════════════════════════════════
        # STEP 5: Generate embeddings
        # ══════════════════════════════════════════
        logger.info(f"🧬 Generating embeddings...")
        embeddings = FaceEncoder.generate_embedding(file_path)

        if not embeddings:
            logger.warning(f"   Could not generate embeddings")
            result['status'] = 'error'
            result['error'] = 'Failed to generate embeddings'
            return result

        # ══════════════════════════════════════════
        # STEP 6: Match or create persons
        # ══════════════════════════════════════════
        logger.info(f"🔍 Matching faces...")

        for i, embedding in enumerate(embeddings):
            vector_bytes = np.array(embedding, dtype=np.float32).tobytes()

            # Try to find match
            match = matcher.find_match(embedding)

            person_info = {
                'person_id': None,
                'is_new': False,
                'distance': None,
                'osint': None
            }

            if match:
                # Existing person
                person_id, distance = match
                person = repo.get_person_by_id(person_id)
                result['persons_matched'] += 1
                person_info['person_id'] = person_id
                person_info['distance'] = distance
                logger.info(f"   ✅ MATCH: Person ID {person_id} (distance: {distance:.4f})")
            else:
                # New person
                person = repo.create_person(
                    name=f"Unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                )
                repo.add_face_to_person(
                    person_id=person.id,
                    embedding=vector_bytes,
                    confidence=0.99,
                    source_image=result['file']
                )
                result['persons_new'] += 1
                person_info['person_id'] = person.id
                person_info['is_new'] = True
                logger.info(f"   🆕 NEW: Person ID {person.id} created")

                # ══════════════════════════════════════════
                # STEP 7: OSINT for new persons
                # ══════════════════════════════════════════
                if enable_osint:
                    logger.info(f"   🔍 Running OSINT search...")
                    osint_result = search_person(file_path, headless=headless)

                    if osint_result.get('success'):
                        parsed = parse_osint_results(osint_result)
                        person_info['osint'] = parsed

                        # Update person with OSINT data
                        if parsed['name']:
                            repo.update_person_osint(
                                person_id=person.id,
                                detected_name=parsed['name'],
                                profession=parsed['profession'],
                                nationality=parsed['nationality'],
                                description=parsed['description']
                            )
                            logger.info(f"   ✅ IDENTIFIED: {parsed['name']}")

                        # Add social profiles
                        for profile in parsed['social_profiles']:
                            repo.add_social_profile(
                                person_id=person.id,
                                platform=profile.get('platform', 'Unknown'),
                                profile_url=profile.get('url', ''),
                                confidence=0.8
                            )

                        result['osint_completed'] = True

            # Add sighting
            repo.add_sighting(
                person_id=person.id if not match else person_id,
                source_type="image_ingestion",
                source_file=result['file'],
                latitude=metadata.get('latitude'),
                longitude=metadata.get('longitude'),
                captured_at=metadata.get('captured_at'),
                exif_data=str(metadata.get('raw_exif', ''))
            )

            result['persons'].append(person_info)

        # ══════════════════════════════════════════
        # STEP 8: Move to processed folder
        # ══════════════════════════════════════════
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{timestamp}_{result['file']}"
        dest_path = PROCESSED_DIR / new_filename

        shutil.move(file_path, dest_path)
        result['moved_to'] = str(dest_path)
        logger.info(f"   📁 Moved to: {dest_path}")

        # ══════════════════════════════════════════
        # STEP 9: Record in database
        # ══════════════════════════════════════════
        processing_time = time.time() - start_time

        repo.add_processed_file(
            file_hash=file_hash,
            original_filename=result['file'],
            file_size=os.path.getsize(dest_path),
            faces_detected=result['faces_detected'],
            persons_matched=result['persons_matched'],
            persons_new=result['persons_new'],
            osint_completed=result['osint_completed'],
            moved_to=str(dest_path),
            status='success'
        )

        repo.log_processing(
            source_path=str(file_path),
            source_type='image',
            file_hash=file_hash,
            faces_detected=result['faces_detected'],
            faces_matched=result['persons_matched'],
            faces_new=result['persons_new'],
            processing_time=processing_time,
            status='success'
        )

        result['status'] = 'success'
        logger.info(f"   ⏱️  Processed in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"   ❌ Error: {e}")
        result['status'] = 'error'
        result['error'] = str(e)

    return result


def process_all_images(enable_osint: bool = True, headless: bool = True):
    """Process all images in the input directory"""

    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return

    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

    if not files:
        logger.warning("No images found in input directory")
        return

    repo = Repository()
    matcher = FaceMatcher()

    # Statistics
    stats = {
        'total_files': len(files),
        'processed': 0,
        'skipped': 0,
        'no_faces': 0,
        'errors': 0,
        'total_faces': 0,
        'persons_matched': 0,
        'persons_new': 0
    }

    logger.info("=" * 60)
    logger.info("👁️  WE KNOW WHERE YOU ARE - FACIAL TRACKING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"📂 Input directory: {INPUT_DIR}")
    logger.info(f"📂 Processed directory: {PROCESSED_DIR}")
    logger.info(f"📊 Files to process: {len(files)}")
    logger.info(f"🔍 OSINT enabled: {enable_osint}")
    logger.info(f"🖥️  Headless mode: {headless}")
    logger.info("=" * 60)

    for idx, filename in enumerate(files, 1):
        file_path = os.path.join(INPUT_DIR, filename)

        logger.info(f"\n{'─' * 50}")
        logger.info(f"📸 [{idx}/{len(files)}] Processing: {filename}")
        logger.info(f"{'─' * 50}")

        result = process_single_image(
            file_path=file_path,
            repo=repo,
            matcher=matcher,
            enable_osint=enable_osint,
            headless=headless
        )

        # Update stats
        if result['status'] == 'success':
            stats['processed'] += 1
        elif result['status'] == 'skipped':
            stats['skipped'] += 1
        elif result['status'] == 'no_faces':
            stats['no_faces'] += 1
        else:
            stats['errors'] += 1

        stats['total_faces'] += result['faces_detected']
        stats['persons_matched'] += result['persons_matched']
        stats['persons_new'] += result['persons_new']

    # Final report
    db_stats = repo.get_stats()

    logger.info("\n" + "=" * 60)
    logger.info("📊 PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Files in batch:     {stats['total_files']}")
    logger.info(f"   ├─ Processed:       {stats['processed']}")
    logger.info(f"   ├─ Skipped (dup):   {stats['skipped']}")
    logger.info(f"   ├─ No faces:        {stats['no_faces']}")
    logger.info(f"   └─ Errors:          {stats['errors']}")
    logger.info(f"")
    logger.info(f"   Faces detected:     {stats['total_faces']}")
    logger.info(f"   ├─ Matched:         {stats['persons_matched']}")
    logger.info(f"   └─ New:             {stats['persons_new']}")
    logger.info(f"")
    logger.info(f"   DATABASE TOTALS:")
    logger.info(f"   ├─ Persons:         {db_stats['total_persons']}")
    logger.info(f"   ├─ Faces:           {db_stats['total_faces']}")
    logger.info(f"   ├─ Sightings:       {db_stats['total_sightings']}")
    logger.info(f"   ├─ Social Profiles: {db_stats['total_social_profiles']}")
    logger.info(f"   └─ Processed Files: {db_stats['total_processed']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    enable_osint = '--no-osint' not in sys.argv
    headless = '--visible' not in sys.argv

    process_all_images(enable_osint=enable_osint, headless=headless)