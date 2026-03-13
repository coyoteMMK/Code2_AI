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

const publicBasePath = process.env.NODE_ENV === "production" ? "/Code2_AI" : "";

export const metadata = {
  title: "Code2 AI By Santos Bernabeu",
  description: "Convertidor de instrucciones en código ensamblador medianrte IA",
  icons: {
    icon: [{ url: "/code2-logo.svg", type: "image/svg+xml" }],
    shortcut: ["/code2-logo.svg"],
    apple: [{ url: "/code2-logo.svg" }],
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href={`${publicBasePath}/code2-logo.svg`} type="image/svg+xml" />
        <link rel="shortcut icon" href={`${publicBasePath}/code2-logo.svg`} type="image/svg+xml" />
        <link rel="apple-touch-icon" href={`${publicBasePath}/code2-logo.svg`} />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
