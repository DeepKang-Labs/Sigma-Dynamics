import { Card, CardContent } from "@/components/ui/card"; import { ArrowDown } from "lucide-react"; import { motion } from "framer-motion";

export default function SigmaInfluenceFlow() { const blocks = [ { title: "MONDE RÉEL", description: "Marchés, hashrate, réseaux, société", color: "from-cyan-500 to-blue-700", }, { title: "DONNÉES COLLECTÉES", description: "Prix BTC, SKY, SCH, hashrate, volumes, liquidité, indicateurs externes (énergie, météo…)", color: "from-blue-700 to-indigo-700", }, { title: "CHAMP Σ-DYNAMICS", description: "Génère les états internes : Θ, V, Σ → mesure de cohérence et d’équilibre", color: "from-indigo-700 to-purple-700", }, { title: "Σ-ANALYSIS", description: "Analyse les métriques : cohérence, veto, résilience → produit une lecture du climat global", color: "from-purple-700 to-pink-600", }, { title: "DÉCISION / ACTION HUMAINE", description: "Ajustement des carnets d’ordre, communication, publication, choix éthique ou économique", color: "from-pink-600 to-rose-600", }, { title: "MARCHÉ / ÉCOSYSTÈME", description: "Les actions humaines modifient : prix, flux, confiance, consensus", color: "from-rose-600 to-cyan-500", }, ];

return ( <div className="flex flex-col items-center space-y-4 p-6 bg-gradient-to-b from-gray-950 to-black min-h-screen text-white overflow-hidden"> <h1 className="text-3xl font-bold text-cyan-400 mb-6">Boucle d’influence Sigma</h1> {blocks.map((block, i) => ( <motion.div key={i} initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.2 }} className="w-full max-w-3xl" > <Card className={bg-gradient-to-r ${block.color} text-white rounded-2xl shadow-lg shadow-cyan-500/20}> <CardContent className="p-6"> <h2 className="text-xl font-semibold mb-2">{block.title}</h2> <p className="text-sm text-gray-100">{block.description}</p> </CardContent> </Card> {i < blocks.length - 1 && ( <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.25 }} className="flex justify-center my-2" > <ArrowDown className="text-cyan-400 animate-bounce" size={30} /> </motion.div> )} </motion.div> ))}

{/* Cercle fractal Σ animé */}
  <motion.div
    initial={{ rotate: 0, scale: 0.8, opacity: 0.6 }}
    animate={{ rotate: 360, scale: 1, opacity: 1 }}
    transition={{ repeat: Infinity, duration: 12, ease: "linear" }}
    className="mt-10 w-40 h-40 rounded-full border-4 border-cyan-400 shadow-[0_0_30px_rgba(6,182,212,0.5)] flex items-center justify-center"
  >
    <span className="text-5xl font-bold text-cyan-300">Σ</span>
  </motion.div>

  <div className="mt-6 text-center text-cyan-300 text-sm">
    ⇦ Boucle de rétroaction Σ ⇨  —  Le champ observe, analyse, inspire et rééquilibre le monde.
  </div>
</div>

); }

