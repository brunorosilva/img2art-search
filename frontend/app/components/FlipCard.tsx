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
      className="mx-2 flex-shrink-0 cursor-pointer group overflow-hidden"
      style={{
        width: '250px',
        aspectRatio: '5 / 4',
        perspective: '1000px',
      }}
    >
      <div
        className="relative w-full h-full transition-transform duration-700 ease-out group-hover:[transform:rotateY(180deg)]"
        style={{
          transformStyle: 'preserve-3d',
          WebkitTransformStyle: 'preserve-3d',
          willChange: 'transform',
        }}
      >
        {/* Front - Photo */}
        <div
          className="absolute inset-0 rounded-xl overflow-hidden"
          style={{
            backfaceVisibility: 'hidden',
            WebkitBackfaceVisibility: 'hidden',
            transform: 'rotateY(0deg)',
          }}
        >
          <Image
            src={photoUrl}
            alt="Photo"
            fill
            className="object-cover"
            sizes="250px"
            unoptimized
          />
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
            sizes="250px"
            unoptimized
          />
        </div>
      </div>
    </div>
  );
}
