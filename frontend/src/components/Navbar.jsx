export default function Navbar() {
  return (
    <nav className="border-b border-[#1F2937] px-6 py-4">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🚢</span>
          <span className="font-bold text-white text-lg tracking-tight">
            Titanic <span className="text-[#60A5FA]">Predictor</span>
          </span>
        </div>
        <div className="flex items-center gap-6 text-sm text-[#6B7280]">
          <a
            href="https://github.com/Bhumik2005"
            target="_blank"
            rel="noreferrer"
            className="hover:text-white transition-colors"
          >
            GitHub
          </a>
          <a
            href="https://portfolio-mauve-three-ni0ea0fq29.vercel.app"
            target="_blank"
            rel="noreferrer"
            className="hover:text-white transition-colors"
          >
            Portfolio
          </a>
        </div>
      </div>
    </nav>
  )
}
