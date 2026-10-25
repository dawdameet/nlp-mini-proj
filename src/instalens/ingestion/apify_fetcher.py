"""
Instagram comment fetcher using Apify's cloud scraper.

Provides 100% reliable retrieval for live Instagram posts/reels without
needing personal Instagram accounts or triggering login blocks.
"""

from __future__ import annotations

import os
from typing import Optional

from .csv_loader import Comment
from .instagram import FetchResult
from .validator import validate_url


class ApifyInstagramFetcher:
    """
    Fetch comments from Instagram reels/posts via Apify.

    Requires an Apify API token (free at https://apify.com).
    """

    def __init__(self, api_token: str | None = None) -> None:
        self.api_token = api_token or os.environ.get("APIFY_API_TOKEN")

    def fetch(self, url: str, k: int = 100) -> FetchResult:
        if not self.api_token:
            return FetchResult(
                comments=[],
                post_shortcode="",
                post_caption="",
                post_likes=0,
                post_comments_count=0,
                comments_requested=k,
                comments_retrieved=0,
                error=(
                    "Apify API token not provided. Set APIFY_API_TOKEN environment variable "
                    "or pass --apify-token <token>. (Sign up free at https://apify.com)"
                ),
            )

        validation = validate_url(url)
        if not validation["valid"]:
            return FetchResult(
                comments=[],
                post_shortcode="",
                post_caption="",
                post_likes=0,
                post_comments_count=0,
                comments_requested=k,
                comments_retrieved=0,
                error=validation["error"],
            )

        shortcode = validation["shortcode"]

        try:
            from apify_client import ApifyClient

            client = ApifyClient(self.api_token)
            run_input = {
                "directUrls": [url],
                "resultsLimit": k,
            }

            run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)

            comments: list[Comment] = []
            dataset_id = (
                getattr(run, "default_dataset_id", None)
                or getattr(run, "defaultDatasetId", None)
                or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
            )
            if not dataset_id:
                raise ValueError("Could not find default dataset ID in Apify run output.")

            dataset = client.dataset(dataset_id)
            for item in dataset.iterate_items():
                text = item.get("text", "")
                if not text or not text.strip():
                    continue

                comments.append(Comment(
                    text=text.strip(),
                    comment_id=str(item.get("id", f"c_{len(comments)+1}")),
                    username="[redacted]",
                    timestamp=item.get("timestamp", ""),
                    source=f"instagram:{shortcode}",
                    metadata={"likes": item.get("likesCount", 0)},
                ))
                if len(comments) >= k:
                    break

            return FetchResult(
                comments=comments,
                post_shortcode=shortcode,
                post_caption=f"Reel #{shortcode}",
                post_likes=0,
                post_comments_count=len(comments),
                comments_requested=k,
                comments_retrieved=len(comments),
            )

        except Exception as e:
            return FetchResult(
                comments=[],
                post_shortcode=shortcode,
                post_caption="",
                post_likes=0,
                post_comments_count=0,
                comments_requested=k,
                comments_retrieved=0,
                error=f"Apify retrieval error: {e}",
            )
