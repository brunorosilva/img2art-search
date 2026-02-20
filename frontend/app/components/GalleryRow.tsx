'use client';

import FlipCard from './FlipCard';

interface CardData {
  photoUrl: string;
  artworkUrl: string;
  artworkTitle?: string;
  artist?: string;
}

interface GalleryRowProps {
  cards: CardData[];
  direction: 'left' | 'right';
  speed: 'slow' | 'normal' | 'fast';
}

export default function GalleryRow({ cards, direction, speed }: GalleryRowProps) {
  const animationClass = direction === 'left'
    ? speed === 'slow' ? 'animate-scroll-left-slow' : speed === 'fast' ? 'animate-scroll-left-fast' : 'animate-scroll-left'
    : speed === 'slow' ? 'animate-scroll-right-slow' : speed === 'fast' ? 'animate-scroll-right-fast' : 'animate-scroll-right';

  // Double the cards for seamless loop
  const doubledCards = [...cards, ...cards];

  return (
    <div className="gallery-row overflow-hidden h-full">
      <div className={`flex items-center h-full ${animationClass}`} style={{ width: 'fit-content' }}>
        {doubledCards.map((card, index) => (
          <FlipCard
            key={`${index}-${card.photoUrl}`}
            photoUrl={card.photoUrl}
            artworkUrl={card.artworkUrl}
            artworkTitle={card.artworkTitle}
            artist={card.artist}
          />
        ))}
      </div>
    </div>
  );
}
