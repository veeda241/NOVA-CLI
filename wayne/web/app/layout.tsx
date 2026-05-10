import type { Metadata } from "next";
import { Orbitron, Share_Tech_Mono } from "next/font/google";
import WakeListener from "../components/WakeListener";
import "./globals.css";

const orbitron = Orbitron({ subsets: ["latin"], variable: "--font-orbitron" });
const shareTech = Share_Tech_Mono({ subsets: ["latin"], weight: "400", variable: "--font-share-tech" });

export const metadata: Metadata = {
  title: "W.A.Y.N.E — Wireless Artificial Yielding Network Engine",
  description: "Wireless Artificial Yielding Network Engine console"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${orbitron.variable} ${shareTech.variable} font-body`}>
        <WakeListener />
        {children}
      </body>
    </html>
  );
}
