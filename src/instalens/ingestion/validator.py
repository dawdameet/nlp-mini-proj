"""
Instagram URL validation.

Validates supported Instagram post URL formats.
Does NOT verify that the post is accessible — that's the adapter's job.
"""

import re
from typing import Optional


# Supported URL patterns
_INSTAGRAM_POST_RE = re.compile(
    r"^https?://(?:www\.)?instagram\.com/(?:share/)?(p|reel|reels|tv)/([A-Za-z0-9_-]{9,15})/?(?:[?#].*)?$",
    re.IGNORECASE,
)

_INSTAGRAM_SHORTCODE_RE = re.compile(
    r"^[A-Za-z0-9_-]{9,15}$"
)

SUPPORTED_TYPES = {"p", "reel", "reels", "tv"}


def validate_url(url: str) -> dict:
    """
    Validate an Instagram post URL.

    Parameters
    ----------
    url : str
        The URL to validate.

    Returns
    -------
    dict
        {
            "valid": bool,
            "shortcode": str | None,
            "media_type": str | None,  # "p", "reel", or "tv"
            "error": str | None,
        }
    """
    # Clean shell-escaped characters (e.g. zsh auto-escaping \? and \=)
    url = url.strip().replace("\\", "")

    if not url:
        return {"valid": False, "shortcode": None, "media_type": None,
                "error": "Empty URL"}

    # Try full URL match
    match = _INSTAGRAM_POST_RE.match(url)
    if match:
        media_type = match.group(1).lower()
        if media_type == "reels":
            media_type = "reel"
        shortcode = match.group(2)
        return {
            "valid": True,
            "shortcode": shortcode,
            "media_type": media_type,
            "error": None,
        }

    # Try bare shortcode
    if _INSTAGRAM_SHORTCODE_RE.match(url):
        return {
            "valid": True,
            "shortcode": url,
            "media_type": "unknown",
            "error": None,
        }

    # Check if it's an Instagram URL but unsupported format
    if "instagram.com" in url.lower():
        return {
            "valid": False,
            "shortcode": None,
            "media_type": None,
            "error": (
                "Unsupported Instagram URL format. "
                f"Supported types: {', '.join(SUPPORTED_TYPES)}. "
                "Example: https://www.instagram.com/p/ABC12345678/"
            ),
        }

    return {
        "valid": False,
        "shortcode": None,
        "media_type": None,
        "error": "Not a recognized Instagram URL.",
    }
