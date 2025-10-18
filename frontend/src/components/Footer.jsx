import React from "react";

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-200 mt-16">
      <div className="max-w-7xl mx-auto px-6 py-12 grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
          <div className="flex items-center">
            <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center text-red-600 font-serif">P</div>
            <h3 className="ml-3 text-lg font-semibold">Predict food</h3>
          </div>
          <p className="text-sm text-gray-400 mt-4 max-w-xs">
            Discover calories fast and easily. Scan, predict and learn more about your daily food intake.
          </p>
        </div>

        <div className="flex justify-between md:justify-center">
          <div>
            <h4 className="font-semibold mb-3">Product</h4>
            <ul className="text-sm text-gray-400 space-y-2">
              <li><a href="#features" className="hover:text-white">Features</a></li>
              <li><a href="#pricing" className="hover:text-white">Pricing</a></li>
              <li><a href="#download" className="hover:text-white">Download</a></li>
            </ul>
          </div>

          <div className="ml-8">
            <h4 className="font-semibold mb-3">Company</h4>
            <ul className="text-sm text-gray-400 space-y-2">
              <li><a href="#about" className="hover:text-white">About</a></li>
              <li><a href="#careers" className="hover:text-white">Careers</a></li>
              <li><a href="#contact" className="hover:text-white">Contact</a></li>
            </ul>
          </div>
        </div>

        <div>
          <h4 className="font-semibold mb-3">Newsletter</h4>
          <p className="text-sm text-gray-400 mb-4">Subscribe to get updates and useful tips.</p>
          <div className="flex">
            <input type="email" placeholder="Your email" className="p-2 rounded-l-md border-0 text-gray-800" />
            <button className="px-4 rounded-r-md bg-red-600 text-white">Subscribe</button>
          </div>

          <div className="flex items-center gap-3 mt-6">
            <a href="#" className="w-9 h-9 rounded-full bg-white/8 flex items-center justify-center hover:bg-white/12">
              {/* twitter */}
              <svg className="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M..." /></svg>
            </a>
            <a href="#" className="w-9 h-9 rounded-full bg-white/8 flex items-center justify-center hover:bg-white/12">
              <svg className="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M..." /></svg>
            </a>
            <a href="#" className="w-9 h-9 rounded-full bg-white/8 flex items-center justify-center hover:bg-white/12">
              <svg className="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M..." /></svg>
            </a>
          </div>
        </div>
      </div>

      <div className="border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4 text-sm text-gray-500 flex flex-col md:flex-row justify-between">
          <span>© {new Date().getFullYear()} Predict Food | Proyecto realizado por Maria, Yeder, Cristian y Lady para el Bootcamp IA (Factoría F5).
              </span>
          <div className="mt-2 md:mt-0">
            <a href="#" className="hover:text-white mx-2">Privacy</a>
            <a href="#" className="hover:text-white mx-2">Terms</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
