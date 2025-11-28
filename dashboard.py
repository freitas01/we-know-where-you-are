"""
WE KNOW WHERE YOU ARE - Facial Tracking Dashboard
Dashboard de Rastreamento Facial e OSINT
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import webbrowser
import base64
import hashlib
from pathlib import Path
from sqlalchemy.orm import joinedload

from src.database.repository import Repository
from src.database.models import Person, Sighting, SocialProfile, Face
from src.face_engine.detector import FaceDetector
from src.face_engine.encoder import FaceEncoder
from src.face_engine.matcher import FaceMatcher
from src.metadata.extractor import MetadataExtractor
from src.osint.yandex_search import search_person
from src.config import INPUT_DIR, DATA_DIR

import numpy as np
from datetime import datetime

PROCESSED_DIR = DATA_DIR / "processed"

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="We Know Where You Are",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-top: 0;
    }
    .person-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #0f3460;
        color: #ffffff;
    }
    .person-card h3 {
        color: #e94560;
        margin-bottom: 15px;
        font-size: 1.4rem;
    }
    .person-card .info-row {
        display: flex;
        margin: 8px 0;
        font-size: 14px;
    }
    .person-card .label {
        color: #a0a0a0;
        min-width: 120px;
    }
    .person-card .value {
        color: #ffffff;
        font-weight: 500;
    }
    .identified {
        background: linear-gradient(135deg, #0a3d0a 0%, #1a5c1a 100%);
        border: 1px solid #2d8a2d;
    }
    .identified h3 {
        color: #4ade80;
    }
    .social-badge {
        display: inline-block;
        background: #0f3460;
        color: #fff;
        padding: 4px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 12px;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
    }
    .osint-result {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4ade80;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def calculate_sha256(file_bytes):
    """Calculate SHA256 hash of file bytes"""
    return hashlib.sha256(file_bytes).hexdigest()

def process_uploaded_files(uploaded_files, enable_osint=True):
    """Process uploaded images through the facial recognition pipeline"""
    from src.osint.yandex_search import search_person
    import shutil

    repo = Repository()
    matcher = FaceMatcher()
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()
        file_hash = calculate_sha256(file_bytes)

        # Check for duplicate
        if repo.is_file_processed(file_hash):
            results.append({
                'file': uploaded_file.name,
                'status': 'skipped',
                'message': 'File already processed (duplicate SHA256)'
            })
            progress_bar.progress((idx + 1) / len(uploaded_files))
            continue

        status_text.text(f"🔍 Processing: {uploaded_file.name}")

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Extract metadata
            metadata = MetadataExtractor.extract(tmp_path)

            # Detect faces
            status_text.text(f"👤 Detecting faces in: {uploaded_file.name}")
            faces = FaceDetector.detect_faces(tmp_path)

            if not faces:
                results.append({
                    'file': uploaded_file.name,
                    'status': 'no_faces',
                    'message': 'No faces detected'
                })
                repo.add_processed_file(
                    file_hash=file_hash,
                    original_filename=uploaded_file.name,
                    file_size=len(file_bytes),
                    status='no_faces'
                )
                continue

            # Generate embeddings
            status_text.text(f"🧬 Generating embeddings: {uploaded_file.name}")
            embeddings = FaceEncoder.generate_embedding(tmp_path)

            file_results = {
                'file': uploaded_file.name,
                'status': 'success',
                'faces': len(faces),
                'persons': [],
                'metadata': metadata
            }

            persons_matched = 0
            persons_new = 0

            for i, embedding in enumerate(embeddings):
                vector_bytes = np.array(embedding, dtype=np.float32).tobytes()

                # Try to match
                match = matcher.find_match(embedding)

                if match:
                    person_id, distance = match
                    person = repo.get_person_by_id(person_id)
                    is_new = False
                    persons_matched += 1
                else:
                    person = repo.create_person(
                        name=f"Unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                    )
                    repo.add_face_to_person(
                        person_id=person.id,
                        embedding=vector_bytes,
                        confidence=0.99,
                        source_image=uploaded_file.name
                    )
                    is_new = True
                    persons_new += 1

                    # OSINT search for new persons
                    if enable_osint:
                        status_text.text(f"🔍 OSINT search for: {uploaded_file.name}")
                        osint_result = search_person(tmp_path, headless=True)

                        if osint_result.get('success') and osint_result.get('raw_text'):
                            raw_text = osint_result['raw_text'].lower()

                            # Try to identify
                            detected_name = None
                            profession = None
                            nationality = None

                            if 'neymar' in raw_text:
                                detected_name = 'Neymar Jr'
                                profession = 'Professional Footballer'
                                nationality = 'Brazilian'
                            elif 'messi' in raw_text:
                                detected_name = 'Lionel Messi'
                                profession = 'Professional Footballer'
                                nationality = 'Argentine'
                            elif 'ronaldo' in raw_text:
                                detected_name = 'Cristiano Ronaldo'
                                profession = 'Professional Footballer'
                                nationality = 'Portuguese'
                            elif 'lula' in raw_text or 'луис' in raw_text.lower():
                                detected_name = 'Luiz Inácio Lula da Silva'
                                profession = 'President of Brazil'
                                nationality = 'Brazilian'
                            elif 'bolsonaro' in raw_text:
                                detected_name = 'Jair Bolsonaro'
                                profession = 'Former President of Brazil'
                                nationality = 'Brazilian'
                            elif 'elon' in raw_text or 'musk' in raw_text:
                                detected_name = 'Elon Musk'
                                profession = 'CEO Tesla/SpaceX'
                                nationality = 'American'

                            if detected_name:
                                repo.update_person_osint(
                                    person_id=person.id,
                                    detected_name=detected_name,
                                    profession=profession,
                                    nationality=nationality
                                )

                            # Add social profiles if found
                            for profile in osint_result.get('social_profiles', []):
                                repo.add_social_profile(
                                    person_id=person.id,
                                    platform=profile.get('platform', 'Unknown'),
                                    profile_url=profile.get('url', ''),
                                    confidence=0.8
                                )

                # Add sighting
                repo.add_sighting(
                    person_id=person.id,
                    source_type="dashboard_upload",
                    source_file=uploaded_file.name,
                    latitude=metadata.get('latitude'),
                    longitude=metadata.get('longitude'),
                    captured_at=metadata.get('captured_at')
                )

                file_results['persons'].append({
                    'id': person.id,
                    'is_new': is_new,
                    'distance': match[1] if match else None
                })

            # Save to input folder
            dest_path = INPUT_DIR / uploaded_file.name
            with open(dest_path, 'wb') as f:
                f.write(file_bytes)

            # Record processed file
            repo.add_processed_file(
                file_hash=file_hash,
                original_filename=uploaded_file.name,
                file_size=len(file_bytes),
                faces_detected=len(faces),
                persons_matched=persons_matched,
                persons_new=persons_new,
                osint_completed=enable_osint,
                status='success'
            )

            results.append(file_results)

        except Exception as e:
            results.append({
                'file': uploaded_file.name,
                'status': 'error',
                'message': str(e)
            })
        finally:
            os.unlink(tmp_path)

        progress_bar.progress((idx + 1) / len(uploaded_files))

    status_text.text("✅ Processing complete!")
    return results

def load_persons_data():
    """Load all persons with their sightings and OSINT data"""
    repo = Repository()
    stats = repo.get_stats()
    session = repo.get_session()

    try:
        persons = session.query(Person).options(
            joinedload(Person.sightings),
            joinedload(Person.faces),
            joinedload(Person.social_profiles)
        ).all()

        persons_data = []
        for p in persons:
            sightings_with_gps = [s for s in p.sightings if s.latitude and s.longitude]

            # Get source image
            source_image = None
            if p.faces:
                source_image = p.faces[0].source_image

            persons_data.append({
                'id': p.id,
                'uuid': p.unique_id,
                'name': p.name,
                'detected_name': p.detected_name,
                'profession': p.profession,
                'nationality': p.nationality,
                'description': p.description,
                'first_seen': p.first_seen,
                'last_seen': p.last_seen,
                'total_sightings': p.total_sightings,
                'sightings': p.sightings,
                'sightings_with_gps': sightings_with_gps,
                'source_image': source_image,
                'social_profiles': p.social_profiles
            })

        return stats, persons_data
    finally:
        session.close()

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

# Header
st.markdown('<p class="main-header">👁️ WE KNOW WHERE YOU ARE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Facial Tracking & OSINT System | Sistema de Rastreamento Facial</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Control Panel")

# Load data
try:
    stats, persons_data = load_persons_data()

    # Stats in sidebar
    st.sidebar.markdown("### 📊 System Statistics")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("👤 Persons", stats['total_persons'])
    col2.metric("👁️ Faces", stats['total_faces'])

    col3, col4 = st.sidebar.columns(2)
    col3.metric("📍 Sightings", stats['total_sightings'])
    col4.metric("🔗 Profiles", stats['total_social_profiles'])

    st.sidebar.metric("📁 Processed", stats['total_processed'])

    st.sidebar.success("✅ Database Connected")

except Exception as e:
    st.sidebar.error(f"❌ Database Error: {e}")
    stats = {'total_persons': 0, 'total_faces': 0, 'total_sightings': 0, 'total_social_profiles': 0, 'total_processed': 0}
    persons_data = []

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
enable_osint = st.sidebar.checkbox("🔍 Enable OSINT", value=True, help="Automatically search for person identity")

# ══════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Process", "👥 Persons Database", "🗺️ Location Map", "🔍 OSINT Search"])

# ══════════════════════════════════════════════════════════════
# TAB 1: UPLOAD & PROCESS
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("📁 Upload Images for Processing")
    st.markdown("Upload images to detect faces, extract metadata, identify persons via OSINT, and track locations.")

    uploaded_files = st.file_uploader(
        "Select images to process",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Drag and drop or click to select multiple images"
    )

    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected")

        col1, col2 = st.columns([1, 3])

        with col1:
            process_btn = st.button("🚀 Process All", type="primary", use_container_width=True)

        if process_btn:
            with st.spinner("Processing images... This may take a while if OSINT is enabled."):
                results = process_uploaded_files(uploaded_files, enable_osint=enable_osint)

            st.success("✅ Processing Complete!")

            # Show results
            for result in results:
                if result.get('status') == 'success':
                    with st.expander(f"✅ {result['file']} - {result.get('faces', 0)} face(s)", expanded=True):
                        for person in result.get('persons', []):
                            if person['is_new']:
                                st.write(f"  🆕 NEW Person ID: {person['id']}")
                            else:
                                st.write(f"  ✅ MATCH Person ID: {person['id']} (distance: {person['distance']:.4f})")

                        meta = result.get('metadata', {})
                        if meta.get('has_gps'):
                            st.write(f"  📍 GPS: {meta['latitude']:.6f}, {meta['longitude']:.6f}")
                        if meta.get('captured_at'):
                            st.write(f"  📅 Date: {meta['captured_at']}")
                elif result.get('status') == 'skipped':
                    st.warning(f"⏭️ {result['file']} - {result.get('message', 'Skipped')}")
                elif result.get('status') == 'no_faces':
                    st.info(f"👤 {result['file']} - No faces detected")
                else:
                    st.error(f"❌ {result['file']} - {result.get('message', 'Error')}")

            st.balloons()
            st.rerun()

# ══════════════════════════════════════════════════════════════
# TAB 2: PERSONS DATABASE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("👥 Tracked Persons Database")

    if not persons_data:
        st.info("No persons in database. Upload images to start tracking.")
    else:
        # Search/filter
        search = st.text_input("🔍 Search by name or ID", "")

        # Filter identified vs unidentified
        filter_option = st.radio(
            "Filter:",
            ["All", "Identified (OSINT)", "Unidentified"],
            horizontal=True
        )

        # Display persons as cards
        for person in persons_data:
            # Apply filters
            if search:
                search_lower = search.lower()
                if (search_lower not in (person.get('name') or '').lower() and
                        search_lower not in (person.get('detected_name') or '').lower() and
                        search not in str(person['id'])):
                    continue

            if filter_option == "Identified (OSINT)" and not person.get('detected_name'):
                continue
            if filter_option == "Unidentified" and person.get('detected_name'):
                continue

            # Determine if identified
            is_identified = bool(person.get('detected_name'))

            # Create card with columns
            col1, col2 = st.columns([2, 1])

            with col1:
                # Card header
                name_display = person.get('detected_name') or person.get('name') or f"Unknown #{person['id']}"

                if is_identified:
                    st.success(f"✅ **{name_display}**")
                else:
                    st.warning(f"❓ **{name_display}**")

                # Info grid
                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.write(f"🆔 **ID:** {person['id']}")
                    st.write(f"🔑 **UUID:** {person['uuid'][:12]}...")
                    if person.get('profession'):
                        st.write(f"💼 **Profession:** {person['profession']}")
                    if person.get('nationality'):
                        st.write(f"🌍 **Nationality:** {person['nationality']}")

                with info_col2:
                    st.write(f"👁️ **Sightings:** {person['total_sightings']}")
                    st.write(f"📍 **Locations:** {len(person['sightings_with_gps'])}")
                    if person['first_seen']:
                        st.write(f"📅 **First seen:** {person['first_seen'].strftime('%d/%m/%Y %H:%M')}")
                    if person['last_seen']:
                        st.write(f"📅 **Last seen:** {person['last_seen'].strftime('%d/%m/%Y %H:%M')}")

                # Social profiles
                if person.get('social_profiles'):
                    profiles_text = " | ".join([f"🔗 {p.platform}" for p in person['social_profiles']])
                    st.write(f"**Profiles:** {profiles_text}")

            with col2:
                # Image display
                if person['source_image']:
                    img_path = INPUT_DIR / person['source_image']
                    if not img_path.exists():
                        for f in PROCESSED_DIR.glob(f"*{person['source_image']}"):
                            img_path = f
                            break

                    if img_path.exists():
                        st.image(str(img_path), width=180)

                # OSINT buttons
                st.write("**🔍 Search Online:**")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("Google", key=f"g_{person['id']}"):
                        webbrowser.open("https://lens.google.com/")
                with btn_col2:
                    if st.button("Yandex", key=f"y_{person['id']}"):
                        webbrowser.open("https://yandex.com/images/")

            st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TAB 3: LOCATION MAP
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🗺️ Sighting Locations Map")

    # Collect all GPS points
    all_locations = []
    for person in persons_data:
        for sighting in person.get('sightings_with_gps', []):
            all_locations.append({
                'person_id': person['id'],
                'person_name': person.get('detected_name') or person.get('name') or f"Unknown #{person['id']}",
                'lat': sighting.latitude,
                'lon': sighting.longitude,
                'date': sighting.captured_at,
                'source': sighting.source_file
            })

    if all_locations:
        try:
            import folium
            from streamlit_folium import st_folium

            # Create map centered on first location
            center_lat = all_locations[0]['lat']
            center_lon = all_locations[0]['lon']

            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

            # Add markers
            for loc in all_locations:
                popup_text = f"""
                <b>{loc['person_name']}</b><br>
                ID: {loc['person_id']}<br>
                Date: {loc['date']}<br>
                Source: {loc['source']}
                """
                folium.Marker(
                    [loc['lat'], loc['lon']],
                    popup=popup_text,
                    icon=folium.Icon(color='red', icon='user')
                ).add_to(m)

            st_folium(m, width=800, height=500)

            st.success(f"📍 Showing {len(all_locations)} location(s) on map")

        except ImportError:
            st.error("Folium not installed. Run: pip install folium streamlit-folium")
    else:
        st.info("📍 No GPS data available yet.")
        st.markdown("""
        **Tip:** Photos from smartphones usually contain GPS data. 
        Camera photos (like Canon T6i) typically don't have GPS unless you use a GPS accessory.
        """)

# ══════════════════════════════════════════════════════════════
# TAB 4: OSINT SEARCH
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("🔍 OSINT - Open Source Intelligence")
    st.markdown("""
    Use reverse image search to find more information about detected persons.
    Upload an image to search across multiple platforms automatically.
    """)

    osint_file = st.file_uploader(
        "Upload image for OSINT search",
        type=['jpg', 'jpeg', 'png'],
        key="osint_uploader"
    )

    if osint_file:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(osint_file, caption="Image for search", width=250)

        with col2:
            st.markdown("### 🔍 Automatic Search")

            if st.button("🚀 Run OSINT Search", type="primary", use_container_width=True):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(osint_file.getvalue())
                    tmp_path = tmp.name

                with st.spinner("🔍 Searching... This may take 30-60 seconds..."):
                    result = search_person(tmp_path, headless=True)

                os.unlink(tmp_path)

                if result.get('success'):
                    st.success("✅ Search Complete!")

                    # Parse results
                    raw_text = result.get('raw_text', '').lower()

                    detected_name = None
                    profession = None
                    nationality = None

                    if 'neymar' in raw_text:
                        detected_name = 'Neymar Jr'
                        profession = 'Professional Footballer'
                        nationality = 'Brazilian'
                    elif 'messi' in raw_text:
                        detected_name = 'Lionel Messi'
                        profession = 'Professional Footballer'
                        nationality = 'Argentine'
                    elif 'ronaldo' in raw_text:
                        detected_name = 'Cristiano Ronaldo'
                        profession = 'Professional Footballer'
                        nationality = 'Portuguese'
                    elif 'lula' in raw_text or 'луис' in raw_text:
                        detected_name = 'Luiz Inácio Lula da Silva'
                        profession = 'President of Brazil'
                        nationality = 'Brazilian'
                    elif 'bolsonaro' in raw_text:
                        detected_name = 'Jair Bolsonaro'
                        profession = 'Former President of Brazil'
                        nationality = 'Brazilian'
                    elif 'elon' in raw_text or 'musk' in raw_text:
                        detected_name = 'Elon Musk'
                        profession = 'CEO Tesla/SpaceX'
                        nationality = 'American'

                    if detected_name:
                        st.markdown(f"""
                        <div class="osint-result">
                            <h3>✅ PERSON IDENTIFIED!</h3>
                            <p><strong>👤 Name:</strong> {detected_name}</p>
                            <p><strong>💼 Profession:</strong> {profession}</p>
                            <p><strong>🌍 Nationality:</strong> {nationality}</p>
                            <p><strong>🔗 Similar Images:</strong> {result.get('similar_images', 0)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Could not identify person automatically.")
                        st.markdown("**Raw Results:**")
                        st.text(result.get('raw_text', 'No text found')[:500])
                else:
                    st.error(f"❌ Search failed: {result.get('error', 'Unknown error')}")

            st.markdown("---")
            st.markdown("### 🌐 Manual Search")
            st.markdown("Click to open search platforms:")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔍 Google Lens", use_container_width=True):
                    webbrowser.open("https://lens.google.com/")
                    st.info("Google Lens opened!")
                if st.button("🔍 TinEye", use_container_width=True):
                    webbrowser.open("https://tineye.com/")
                    st.info("TinEye opened!")
            with col_b:
                if st.button("🔎 Yandex Images", use_container_width=True):
                    webbrowser.open("https://yandex.com/images/")
                    st.info("Yandex opened!")
                if st.button("📷 PimEyes", use_container_width=True):
                    webbrowser.open("https://pimeyes.com/")
                    st.info("PimEyes opened!")

    st.markdown("---")
    st.warning("""
    ⚠️ **Privacy Warning**
    
    These tools demonstrate how easily someone can be identified from a single photo.
    This is for **educational purposes only** to raise awareness about digital privacy.
    """)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>We Know Where You Are</strong> - Facial Tracking & OSINT System</p>
    <p>🎓 Bachelor's Thesis Project - Systems Analysis and Development</p>
    <p>⚠️ For educational and research purposes only</p>
</div>
""", unsafe_allow_html=True)