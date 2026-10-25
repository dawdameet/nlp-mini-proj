"""
Instagram comment fetcher using instaloader.

Retrieves comments from public Instagram posts/reels given a URL.
Handles rate limiting, authentication via session cookies or credentials,
and provides simulated comments for offline/demo modes.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import instaloader

from .csv_loader import Comment
from .validator import validate_url


@dataclass
class FetchResult:
    """Result of an Instagram comment fetch operation."""
    comments: list[Comment]
    post_shortcode: str
    post_caption: str
    post_likes: int
    post_comments_count: int
    comments_requested: int
    comments_retrieved: int
    error: str | None = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# Curated realistic reel comments for demo mode
_DEMO_REEL_COMMENTS = [
    # Positive / Slang / High Engagement
    "nah bro cooked with this one 🔥🔥",
    "absolutely insane visuals!! so talented",
    "underrated reel fr, deserves 1M views",
    "literally crying at the end 😂😭❤️",
    "the transition is so clean bro 🔥",
    "goated creator 🙌 keep going!",
    "w reel w content w life",
    "obsessed with this, watching on repeat ❤️",
    "can't believe how good this turned out 🔥",
    "pure art honestly ✨",
    "you never disappoint, amazing work!",
    "bro has never missed 🔥🔥",
    "stunning cinematography and color grading",
    "this made my entire day so wholesome ❤️",
    "legendary post 🔥",
    "masterpiece right here! proud of you",
    "the ending was unexpectedly brilliant 😂",
    "best reel I've seen all week hands down",
    "so inspiring and beautifully shot ✨",
    "straight heat 🔥🔥🔥",
    # Conversational / Questions / Neutral
    "what song is playing in the background?",
    "what camera did you shoot this on?",
    "drop the location please! looks stunning",
    "tutorial when??? need to learn this",
    "which editing software did you use?",
    "first comment let's gooo",
    "is this shot in 4k 60fps?",
    "tagged my friend @alex to see this",
    "where did you get that jacket from?",
    "part 2 when? waiting eagerly",
    # Mixed / Nuanced
    "great concept but the pacing felt a bit rushed",
    "visually stunning, though the ending was slightly disappointing",
    "not bad at all, but expected a bit more effort",
    "loved the visuals, however the audio was kinda annoying",
    "interesting idea, but the execution could be better",
    "started off amazing but fell off midway",
    "pretty cool tbh, not the best though",
    # Negative / Critical
    "overrated honestly, seen this same idea ten times already",
    "cringe content, unfollowing",
    "fake and completely scripted 💀",
    "waste of 30 seconds of my life",
    "terrible audio mixing, couldn't hear a thing",
    "this is so misleading and annoying 👎",
    "mid at best, don't get the hype at all",
    "disgusting behavior tbh",
    "L take, completely disagree with this",
    "trash editing, hurts my eyes",
]


class InstagramFetcher:
    """
    Fetch comments from Instagram posts and reels using instaloader.

    Supports:
      - Browser session cookie authentication (`session_id`)
      - Saved session files (`session_file` or username)
      - Username & password authentication (`login_user`, `login_pass`)
      - Offline demo mode for presentation and testing
    """

    def __init__(
        self,
        session_id: str | None = None,
        session_file: str | None = None,
        login_user: str | None = None,
        login_pass: str | None = None,
    ) -> None:
        self._loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
        )

        self._logged_in = False

        # 1. Check for session ID (from argument or environment variable)
        sid = session_id or os.environ.get("INSTAGRAM_SESSION_ID")
        if sid:
            sid = sid.strip()
            self._loader.context._session.cookies.set(
                "sessionid", sid, domain=".instagram.com"
            )
            # Instagram sessionid format is: <user_id>%3A... or <user_id>:...
            user_id_str = sid.split("%3A")[0].split(":")[0]
            if user_id_str.isdigit():
                self._loader.context._session.cookies.set(
                    "ds_user_id", user_id_str, domain=".instagram.com"
                )
                try:
                    self._loader.context.user_id = int(user_id_str)
                except ValueError:
                    pass

            # Make sure instaloader context is marked as logged in
            try:
                username = self._loader.test_login()
                self._loader.context.username = username or user_id_str or "session_user"
            except Exception:
                self._loader.context.username = user_id_str or "session_user"

            self._logged_in = True
            return

        # 2. Check for username + password
        user = login_user or os.environ.get("INSTAGRAM_USERNAME")
        password = login_pass or os.environ.get("INSTAGRAM_PASSWORD")

        if user and password:
            try:
                self._loader.login(user, password)
                self._logged_in = True
            except instaloader.exceptions.BadCredentialsException:
                raise ValueError("Invalid Instagram username or password.")
            except instaloader.exceptions.TwoFactorAuthRequiredException:
                raise ValueError(
                    "Two-factor authentication required. Use --session-id with your "
                    "browser session cookie instead to bypass 2FA challenges."
                )
            return

        # 3. Check for saved session file
        if user and not password:
            try:
                self._loader.load_session_from_file(user, filename=session_file)
                self._logged_in = True
            except FileNotFoundError:
                pass

    def fetch(self, url: str, k: int = 100, demo: bool = False) -> FetchResult:
        """
        Fetch up to k comments from an Instagram post/reel URL.

        Parameters
        ----------
        url : str
            Instagram post/reel URL.
        k : int
            Maximum number of comments to retrieve (default: 100).
        demo : bool
            If True, use realistic simulated reel comments (offline demo mode).

        Returns
        -------
        FetchResult
            Contains comments, post metadata, and any errors/warnings.
        """
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

        if demo:
            return self.fetch_demo(shortcode, k=k)

        try:
            return self._fetch_by_shortcode(shortcode, k)
        except instaloader.exceptions.BadResponseException:
            if self._logged_in:
                return self._error_result(
                    shortcode,
                    k,
                    f"Post '{shortcode}' could not be found, or has been deleted/restricted by Instagram.\n"
                    f"  Please verify the URL is accessible in your browser,\n"
                    f"  or run with --demo to analyze simulated reel comments."
                )
            return self._error_result(
                shortcode,
                k,
                "Instagram blocked unauthenticated access (login wall).\n"
                "  To analyze this live reel, choose one of:\n"
                "    • Pass your browser session cookie:  --session-id <cookie>\n"
                "      (or: export INSTAGRAM_SESSION_ID=\"your_session_id\")\n"
                "    • Pass credentials:                  --login USERNAME PASSWORD\n"
                "    • Run with simulated reel comments:  --demo"
            )
        except instaloader.exceptions.ProfileNotExistsException:
            return self._error_result(shortcode, k, "Post not found or has been deleted.")
        except instaloader.exceptions.PrivateProfileNotFollowedException:
            return self._error_result(shortcode, k, "This post belongs to a private profile.")
        except instaloader.exceptions.LoginRequiredException:
            return self._error_result(
                shortcode,
                k,
                "Instagram requires login to access this reel. "
                "Use --session-id or --login to authenticate."
            )
        except instaloader.exceptions.QueryReturnedBadRequestException:
            return self._error_result(
                shortcode, k,
                "Instagram returned a bad request (400). The reel may not exist or require authentication."
            )
        except instaloader.exceptions.ConnectionException as e:
            if "429" in str(e) or "rate" in str(e).lower():
                return self._error_result(shortcode, k, "Rate limited by Instagram. Please wait a few minutes and try again.")
            return self._error_result(shortcode, k, f"Connection error: {e}")
        except Exception as e:
            return self._error_result(shortcode, k, f"Instagram retrieval error: {e}")

    def fetch_demo(self, shortcode: str, k: int = 100) -> FetchResult:
        """
        Generate realistic simulated reel comments for demonstration and offline testing.
        """
        comments: list[Comment] = []
        total_pool = len(_DEMO_REEL_COMMENTS)

        now_str = datetime.now(timezone.utc).isoformat()
        for i in range(min(k, total_pool)):
            text = _DEMO_REEL_COMMENTS[i]
            # Realistic synthetic like counts
            likes = max(0, int(150 - (i * 3.2))) if i < 30 else (i % 7)
            comments.append(Comment(
                text=text,
                comment_id=f"demo_{shortcode}_{i+1:03d}",
                username="[redacted]",
                timestamp=now_str,
                source=f"instagram:reel:{shortcode}",
                metadata={"likes": likes, "is_demo": True},
            ))

        return FetchResult(
            comments=comments,
            post_shortcode=shortcode,
            post_caption=f"Reel #{shortcode} • Trending Reel Showcase",
            post_likes=18420,
            post_comments_count=max(len(comments), 340),
            comments_requested=k,
            comments_retrieved=len(comments),
            warnings=["[DEMO MODE] Using representative reel comments for demonstration."],
        )

    def _fetch_by_shortcode(self, shortcode: str, k: int) -> FetchResult:
        """Fetch comments by post shortcode using instaloader."""
        post = instaloader.Post.from_shortcode(self._loader.context, shortcode)

        caption = post.caption or ""
        likes = post.likes
        total_comments = post.comments

        comments: list[Comment] = []
        warnings: list[str] = []

        try:
            for ig_comment in post.get_comments():
                if len(comments) >= k:
                    break

                comment_text = ig_comment.text
                if not comment_text or not comment_text.strip():
                    continue

                comments.append(Comment(
                    text=comment_text.strip(),
                    comment_id=str(ig_comment.id),
                    username="[redacted]",
                    timestamp=ig_comment.created_at_utc.isoformat() if ig_comment.created_at_utc else "",
                    source="instagram",
                    metadata={
                        "likes": ig_comment.likes_count if hasattr(ig_comment, "likes_count") else 0,
                    },
                ))

        except instaloader.exceptions.QueryReturnedNotFoundException:
            warnings.append("Some comments could not be loaded (may have been deleted).")
        except Exception as e:
            if comments:
                warnings.append(f"Stopped fetching after {len(comments)} comments: {e}")
            else:
                return self._error_result(
                    shortcode,
                    k,
                    f"Instagram's comment API blocked live extraction ({e}).\n"
                    f"  Reel metadata was loaded: \"{caption[:60]}...\" ({likes:,} likes, {total_comments:,} comments).\n"
                    f"  To run sentiment analysis on representative comments for this reel, run:\n"
                    f"    instalens analyze \"https://www.instagram.com/reel/{shortcode}/\" -k {k} --demo"
                )

        if len(comments) < k and total_comments > len(comments):
            warnings.append(
                f"Requested {k} comments but only retrieved {len(comments)} "
                f"(post has {total_comments} total). Some may be restricted."
            )

        return FetchResult(
            comments=comments,
            post_shortcode=shortcode,
            post_caption=caption[:200] + ("..." if len(caption) > 200 else ""),
            post_likes=likes,
            post_comments_count=total_comments,
            comments_requested=k,
            comments_retrieved=len(comments),
            warnings=warnings,
        )

    @staticmethod
    def _error_result(shortcode: str, k: int, error: str) -> FetchResult:
        return FetchResult(
            comments=[],
            post_shortcode=shortcode,
            post_caption="",
            post_likes=0,
            post_comments_count=0,
            comments_requested=k,
            comments_retrieved=0,
            error=error,
        )
