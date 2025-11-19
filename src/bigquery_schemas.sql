CREATE TABLE `automatization-project-478523.spotify_reco.albums` (
  album_id                      STRING,
  album_name                    STRING,
  album_type                    STRING,
  album_release_date            STRING,
  album_release_precision       STRING,
  album_total_tracks            INT64,
  album_label                   STRING,
  album_spotify_url             STRING,
  album_genres                  STRING,      -- ahora STRING: "pop, latin, rock"
  album_image_url               STRING,
  album_available_markets_count INT64,
  ingested_at                   TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);


CREATE TABLE `automatization-project.spotify_reco.tracks` (
  -- Info de álbum (denormalizada para facilidad)
  album_id                STRING,

  -- Info de track
  track_id                STRING,
  track_name              STRING,
  artists                 STRING,   -- "A, B, C"
  spotify_url             STRING,
  preview_url             STRING,
  uri                     STRING,   -- "spotify:track:..."
  duration_ms             INT64,
  track_number            INT64,
  disc_number             INT64,
  explicit                BOOL,
  popularity              INT64,
  isrc                    STRING,
  available_markets_count INT64,

  ingested_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
