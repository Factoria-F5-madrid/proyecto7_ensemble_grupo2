import React from "react";
import hero from "../assets/hero.png"; // coloca hero.jpg en src/assets

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      {/* HERO */}
      <header id="home" className="relative">
        <div
          className="h-screen bg-cover bg-center flex items-center"
          style={{ backgroundImage: `url(${hero})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 to-black/30"></div>

          <div className="container mx-auto px-6 lg:px-20 relative z-10">
            <div className="max-w-3xl text-center mx-auto py-24">
              <h1 className="text-5xl md:text-6xl font-serif text-black leading-tight drop-shadow-lg">
                Best food <span className="block">calories app</span>
              </h1>
              <p className="mt-6 text-lg text-black/90">
                Discover our calorie prediction app
              </p>

              <div className="mt-8 flex items-center justify-center gap-4">
                <a
                  href="#scan"
                  className="px-6 py-3 rounded-full bg-red-600 text-white font-medium shadow-lg hover:bg-red-700 transition"
                >
                  Scan your food
                </a>
                <a
                  href="#browse"
                  className="px-6 py-3 rounded-full bg-white/90 text-gray-800 font-medium shadow hover:brightness-95 transition"
                >
                  Explore Menu
                </a>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* BROWSE CARDS */}
      <main className="container mx-auto px-6 lg:px-20 -mt-24 relative z-10">
        <section id="browse" className="bg-white rounded-xl shadow-lg py-12 px-6">
          <h2 className="text-3xl font-serif text-center mb-10">Browse Our app</h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Card 1 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                {/* icon */}
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 8c3.866 0 7 1.79 7 4v4a1 1 0 01-1 1H6a1 1 0 01-1-1v-4c0-2.21 3.134-4 7-4z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 8V4" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comidas Saludables</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>

            {/* Card 2 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 10h16M4 14h16" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comidas Populares</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>

            {/* Card 3 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M12 2l3 7h-6l3-7zM5 13h14l-1 9H6l-1-9z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comidas Especiales</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>

            {/* Card 4 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M3 7h18M5 7v14h14V7" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comida internacionales</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>
          </div>
        </section>
      </main>

    </div>
  );
}
