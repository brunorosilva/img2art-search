import type { Metadata } from "next";
import { Cormorant_Garamond, Inter } from "next/font/google";
import "./globals.css";

const cormorant = Cormorant_Garamond({
  variable: "--font-serif",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "img2art | Find Art That Matches Your Photos",
  description: "Upload any image and discover visually similar masterpieces from 81,000+ artworks in the WikiArt collection. Powered by Vision Transformer.",
  keywords: ["art", "image search", "WikiArt", "AI", "Vision Transformer", "artwork matching"],
  authors: [{ name: "Bruno Chicelli" }],
  openGraph: {
    title: "img2art | Find Art That Matches Your Photos",
    description: "Upload any image and discover visually similar masterpieces from 81,000+ artworks.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${cormorant.variable} ${inter.variable} antialiased bg-stone-950`}
      >
        {children}
      </body>
    </html>
  );
}
