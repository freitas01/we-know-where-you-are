"""Metadata and EXIF extraction module - extracts GPS, date, camera info from images"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


class MetadataExtractor:
    """Extracts EXIF metadata from images including GPS coordinates"""

    @staticmethod
    def _get_exif_data(image: Image.Image) -> Dict[str, Any]:
        """Extract raw EXIF data from image"""
        exif_data = {}
        try:
            info = image._getexif()
            if info:
                for tag_id, value in info.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
        except Exception:
            pass
        return exif_data

    @staticmethod
    def _get_gps_info(exif_data: Dict) -> Dict[str, Any]:
        """Extract GPS information from EXIF data"""
        gps_info = {}
        if "GPSInfo" in exif_data:
            for key, value in exif_data["GPSInfo"].items():
                tag = GPSTAGS.get(key, key)
                gps_info[tag] = value
        return gps_info

    @staticmethod
    def _convert_to_degrees(value) -> float:
        """Convert GPS coordinates to decimal degrees"""
        try:
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        except (TypeError, IndexError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _parse_datetime(date_str: str) -> Optional[datetime]:
        """Parse EXIF datetime string to datetime object"""
        try:
            return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        except (ValueError, TypeError):
            return None

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        """
        Extract all relevant metadata from an image

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with extracted metadata
        """
        result = {
            "file_name": os.path.basename(image_path),
            "file_size": None,
            "latitude": None,
            "longitude": None,
            "altitude": None,
            "captured_at": None,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "image_width": None,
            "image_height": None,
            "has_gps": False,
            "raw_exif": {}
        }

        try:
            # File size
            result["file_size"] = os.path.getsize(image_path)

            # Open image
            with Image.open(image_path) as img:
                result["image_width"] = img.width
                result["image_height"] = img.height

                # Get EXIF data
                exif_data = cls._get_exif_data(img)

                # Camera info
                result["camera_make"] = exif_data.get("Make", "").strip() if exif_data.get("Make") else None
                result["camera_model"] = exif_data.get("Model", "").strip() if exif_data.get("Model") else None
                result["software"] = exif_data.get("Software", "").strip() if exif_data.get("Software") else None

                # Date/time
                date_str = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
                if date_str:
                    result["captured_at"] = cls._parse_datetime(date_str)

                # GPS data
                gps_info = cls._get_gps_info(exif_data)

                if gps_info:
                    # Latitude
                    if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
                        lat = cls._convert_to_degrees(gps_info["GPSLatitude"])
                        if gps_info["GPSLatitudeRef"] == "S":
                            lat = -lat
                        result["latitude"] = lat

                    # Longitude
                    if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
                        lon = cls._convert_to_degrees(gps_info["GPSLongitude"])
                        if gps_info["GPSLongitudeRef"] == "W":
                            lon = -lon
                        result["longitude"] = lon

                    # Altitude
                    if "GPSAltitude" in gps_info:
                        try:
                            result["altitude"] = float(gps_info["GPSAltitude"])
                        except (TypeError, ValueError):
                            pass

                    result["has_gps"] = result["latitude"] is not None and result["longitude"] is not None

                # Store raw EXIF (excluding binary data)
                for key, value in exif_data.items():
                    if not isinstance(value, bytes) and key != "GPSInfo":
                        result["raw_exif"][key] = str(value)

        except Exception as e:
            result["error"] = str(e)

        return result

    @classmethod
    def get_gps_coordinates(cls, image_path: str) -> Optional[tuple]:
        """
        Quick method to get just GPS coordinates

        Returns:
            Tuple (latitude, longitude) or None if not available
        """
        metadata = cls.extract(image_path)
        if metadata["has_gps"]:
            return (metadata["latitude"], metadata["longitude"])
        return None

    @classmethod
    def format_location_url(cls, latitude: float, longitude: float) -> str:
        """Generate Google Maps URL for coordinates"""
        return f"https://www.google.com/maps?q={latitude},{longitude}"