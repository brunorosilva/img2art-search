'use client';

import Image from 'next/image';

interface FlipCardProps {
  photoUrl: string;
  artworkUrl: string;
  artworkTitle?: string;
  artist?: string;
}

export default function FlipCard({ photoUrl, artworkUrl, artworkTitle, artist }: FlipCardProps) {
  return (
    <div
      className="mx-2 flex-shrink-0 cursor-pointer"
      style={{
        width: '200px',
        height: '280px',
        perspective: '1000px',
      }}
    >
      <div
        className="relative w-full h-full transition-transform duration-500 ease-in-out hover:[transform:rotateY(180deg)]"
        style={{
          transformStyle: 'preserve-3d',
        }}
      >
        {/* Front - Photo */}
        <div
          className="absolute inset-0 rounded-xl overflow-hidden"
          style={{
            backfaceVisibility: 'hidden',
            WebkitBackfaceVisibility: 'hidden',
          }}
        >
          <Image
            src={photoUrl}
            alt="Photo"
            fill
            className="object-cover"
            sizes="200px"
          />
          <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black/60 to-transparent" />
          <div className="absolute bottom-3 left-3 right-3">
            <span className="text-xs text-white/80 font-medium">Photo</span>
          </div>
        </div>

        {/* Back - Artwork */}
        <div
          className="absolute inset-0 rounded-xl overflow-hidden"
          style={{
            backfaceVisibility: 'hidden',
            WebkitBackfaceVisibility: 'hidden',
            transform: 'rotateY(180deg)',
          }}
        >
          <Image
            src={artworkUrl}
            alt={artworkTitle || 'Artwork'}
            fill
            className="object-cover"
            sizes="200px"
          />
          <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-black/80 to-transparent" />
          <div className="absolute bottom-3 left-3 right-3">
            <p className="text-xs text-white font-semibold truncate">{artworkTitle || 'Matched Artwork'}</p>
            {artist && <p className="text-xs text-white/70 truncate">{artist}</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
