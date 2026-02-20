'use client';

import GalleryRow from './GalleryRow';
import galleryJson from '../data/galleryData.json';

// Get basePath for GitHub Pages deployment
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

// Type for gallery items
interface GalleryItem {
  photoUrl: string;
  artworkUrl: string;
  distance?: string;
}

// Helper to prefix local URLs with basePath
function prefixUrl(url: string): string {
  if (url.startsWith('/') && !url.startsWith('//')) {
    return `${basePath}${url}`;
  }
  return url;
}

// Fallback data in case the JSON is empty (before running generate_gallery.py)
const fallbackData: GalleryItem[] = [
  { photoUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=560&fit=crop', artworkUrl: 'https://uploads7.wikiart.org/images/diego-velazquez/a-young-man-self-portrait-1624.jpg' },
  { photoUrl: 'https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=400&h=560&fit=crop', artworkUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Meisje_met_de_parel.jpg/800px-Meisje_met_de_parel.jpg' },
  { photoUrl: 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400&h=560&fit=crop', artworkUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg/800px-Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg' },
  { photoUrl: 'https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400&h=560&fit=crop', artworkUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Cassius_Marcellus_Coolidge_-_Poker_Game_%281894%29.png/1280px-Cassius_Marcellus_Coolidge_-_Poker_Game_%281894%29.png' },
  { photoUrl: 'https://images.unsplash.com/photo-1433086966358-54859d0ed716?w=400&h=560&fit=crop', artworkUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Turner_-_Rain%2C_Steam_and_Speed_-_National_Gallery_file.jpg/1280px-Turner_-_Rain%2C_Steam_and_Speed_-_National_Gallery_file.jpg' },
  { photoUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&h=560&fit=crop', artworkUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Johannes_Vermeer_%281632-1675%29_-_The_Girl_With_The_Pearl_Earring_%281665%29.jpg/800px-Johannes_Vermeer_%281632-1675%29_-_The_Girl_With_The_Pearl_Earring_%281665%29.jpg' },
  { photoUrl: 'https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=400&h=560&fit=crop', artworkUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg' },
  { photoUrl: 'https://images.unsplash.com/photo-1506929562872-bb421503ef21?w=400&h=560&fit=crop', artworkUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Frederic_Leighton_-_Flaming_June_-_1895.jpg/1024px-Frederic_Leighton_-_Flaming_June_-_1895.jpg' },
];

// Use generated data if available, otherwise use fallback
// Prefix URLs with basePath for GitHub Pages
const galleryItems: GalleryItem[] = (galleryJson.items.length > 0
  ? galleryJson.items
  : fallbackData
).map(item => ({
  ...item,
  photoUrl: prefixUrl(item.photoUrl),
  artworkUrl: prefixUrl(item.artworkUrl),
}));

// Split items into rows (8 items per row, repeat to fill 6 rows)
function createRows(items: GalleryItem[], rowCount: number = 6, itemsPerRow: number = 8): GalleryItem[][] {
  const rows: GalleryItem[][] = [];

  for (let i = 0; i < rowCount; i++) {
    const row: GalleryItem[] = [];
    for (let j = 0; j < itemsPerRow; j++) {
      // Cycle through items if we don't have enough
      const itemIndex = (i * itemsPerRow + j) % items.length;
      row.push(items[itemIndex]);
    }
    rows.push(row);
  }

  return rows;
}

const rows = createRows(galleryItems);

export default function GalleryBackground() {
  const directions: ('left' | 'right')[] = ['left', 'right', 'left', 'right', 'left', 'right'];
  const speeds: ('slow' | 'normal' | 'fast')[] = ['normal', 'slow', 'fast', 'normal', 'slow', 'fast'];

  return (
    <div className="fixed inset-0 z-0 overflow-hidden">
      <div className="absolute inset-0 flex flex-col justify-center items-start gap-4 -rotate-6 scale-110" style={{ minHeight: 'max-content' }}>
        {rows.map((rowCards, index) => (
          <div key={index} className="flex-shrink-0" style={{ height: '200px' }}>
            <GalleryRow
              cards={rowCards}
              direction={directions[index]}
              speed={speeds[index]}
            />
          </div>
        ))}
      </div>
      {/* Gradient overlays for depth - pointer-events-none so cards are still interactive */}
      <div className="absolute inset-0 bg-gradient-to-t from-stone-950 via-stone-950/60 to-stone-950 opacity-80 pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-r from-stone-950 via-transparent to-stone-950 opacity-60 pointer-events-none" />
    </div>
  );
}
