import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
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
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-black`}
      >
        {children}
      </body>
    </html>
  );
}
