import React, { useState, useEffect } from "react";
import bowl from "../assets/bowl.png";

const LINKS = [
  { name: "Home", href: "/" },
  { name: "About", href: "/about" },
  { name: "Menu", href: "/menu" },
  { name: "Pages", href: "/pages" },
  { name: "Team", href: "/team" },
];

export default function Nav() {
  const [active, setActive] = useState("/");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") setActive(window.location.pathname || "/");
    // close menu on route change (simple listener)
    const handleRoute = () => setOpen(false);
    window.addEventListener("popstate", handleRoute);
    return () => window.removeEventListener("popstate", handleRoute);
  }, []);

  const onLinkClick = (href) => {
    setActive(href);
    setOpen(false);
  };

  return (
    <header className="w-full bg-[#f7faf5] border-b border-gray-100 sticky top-0 z-50">
      <nav className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="h-16 flex items-center justify-between">
          {/* Left: logo + name */}
          <div className="flex items-center gap-3">
            <img src={bowl} alt="bowl logo" className="w-8 h-8 object-contain" />
            <div className="flex flex-col leading-none">
              <span className="text-2xl font-serif text-gray-800 select-none">Predict food</span>
            </div>
          </div>

          {/* Center: links (desktop) */}
          <div className="hidden md:flex md:items-center md:space-x-6">
            {LINKS.map((link) => {
              const isActive = active === link.href || (link.href === "/" && (active === "" || active === "/"));
              return (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={() => onLinkClick(link.href)}
                  className={
                    "px-3 py-1 rounded-full text-sm font-medium transition-colors " +
                    (isActive
                      ? "bg-[#e6f0df] text-[#1f3a2e] shadow-sm"
                      : "text-gray-600 hover:text-gray-800 hover:bg-gray-100")
                  }
                  aria-current={isActive ? "page" : undefined}
                >
                  {link.name}
                </a>
              );
            })}
          </div>

          {/* Right: sign-in + mobile button */}
          <div className="flex items-center gap-4">
            <a
              href="/login"
              className="hidden sm:inline-flex items-center px-4 py-1.5 border rounded-full text-sm font-medium text-gray-800 border-gray-300 hover:shadow-sm transition"
            >
              Inicia sesión
            </a>

            {/* Mobile menu button */}
            <button
              onClick={() => setOpen(true)}
              type="button"
              className="md:hidden inline-flex items-center justify-center p-2 rounded-md text-gray-600 hover:bg-gray-100"
              aria-expanded={open}
              aria-label="Open menu"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 8h16M4 16h16" />
              </svg>
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Drawer */}
      <div
        className={`fixed inset-0 z-40 transform ${open ? "translate-x-0" : "translate-x-full"} transition-transform duration-300 ease-in-out`}
        aria-hidden={!open}
      >
        {/* backdrop */}
        <button
          className={`absolute inset-0 bg-black/40 ${open ? "opacity-100" : "opacity-0"} transition-opacity`}
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />

        {/* panel */}
        <div className="absolute right-0 top-0 h-full w-[86%] max-w-xs bg-white shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <img src={bowl} alt="bowl" className="w-8 h-8 object-contain" />
              <span className="text-lg font-semibold text-gray-800">Bistro Bliss</span>
            </div>
            <button onClick={() => setOpen(false)} className="p-2 rounded-md text-gray-600 hover:bg-gray-100" aria-label="Close menu">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <nav className="flex flex-col gap-2">
            {LINKS.map((link) => {
              const isActive = active === link.href || (link.href === "/" && (active === "" || active === "/"));
              return (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={() => onLinkClick(link.href)}
                  className={
                    "block px-4 py-2 rounded-md text-base font-medium transition " +
                    (isActive ? "bg-[#e6f0df] text-[#1f3a2e]" : "text-gray-700 hover:bg-gray-100")
                  }
                >
                  {link.name}
                </a>
              );
            })}
          </nav>

          <div className="mt-6">
            <a
              href="/login"
              onClick={() => { setOpen(false); }}
              className="w-full inline-flex items-center justify-center px-4 py-2 rounded-full bg-[#1f3a2e] text-white font-medium"
            >
              Inicia sesión
            </a>
          </div>

          <div className="mt-6 text-sm text-gray-500">
            <p>© {new Date().getFullYear()} Bistro Bliss</p>
          </div>
        </div>
      </div>
    </header>
  );
}
