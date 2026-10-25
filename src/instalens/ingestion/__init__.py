"""Comment ingestion adapters — CSV loader, Instagram fetcher, and URL validation."""

from .csv_loader import CSVCommentLoader
from .validator import validate_url
from .instagram import InstagramFetcher
from .apify_fetcher import ApifyInstagramFetcher

__all__ = ["CSVCommentLoader", "validate_url", "InstagramFetcher", "ApifyInstagramFetcher"]
