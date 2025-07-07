import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Badge } from './ui/badge';
import { BrainCircuit, Star, Zap, Atom, Gem } from 'lucide-react';

type GolemStatsProps = {
  stats: any;
};

const StatItem = ({
  icon,
  label,
  value,
  colorClass,
  progressValue,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  colorClass: string;
  progressValue?: number;
}) => (
  <div className="flex flex-col gap-1">
    <div className="flex items-center justify-between text-xs text-muted-foreground">
      <div className="flex items-center gap-1.5">
        {icon}
        <span>{label}</span>
      </div>
      <span className={`font-mono font-medium ${colorClass}`}>{value}</span>
    </div>
    {progressValue !== undefined && (
      <Progress
        value={progressValue * 100}
        className="h-1.5"
        indicatorClassName={colorClass.replace('text-', 'bg-')}
      />
    )}
  </div>
);

const SefirotDisplay = ({
  sefirot,
  dominant,
}: {
  sefirot: Record<string, number>;
  dominant: string;
}) => {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-foreground">
        Sefirot Activations
      </h4>
      {Object.entries(sefirot).map(([name, value]) => (
        <div key={name} className="flex items-center gap-2">
          <span className="w-20 truncate text-xs text-muted-foreground">
            {name}
          </span>
          <div className="h-2.5 flex-1 rounded-full bg-muted">
            <div
              className={`h-2.5 rounded-full ${
                name === dominant ? 'bg-accent' : 'bg-primary/50'
              }`}
              style={{ width: `${value * 100}%` }}
            />
          </div>
          <span className="w-10 text-right font-mono text-xs text-foreground">
            {(value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
};

export function GolemStats({ stats }: GolemStatsProps) {
  if (!stats) return null;

  const { golem_state, quality_metrics, golem_analysis, aether_data } = stats;

  if (!golem_state || !quality_metrics || !golem_analysis || !aether_data) {
    return (
      <div className="mt-2 text-xs text-card-foreground/80">
        <p>Incomplete Golem stats received.</p>
        <details>
          <summary>Show raw data</summary>
          <pre className="mt-2 overflow-x-auto rounded-md bg-card/50 p-2 text-xs">
            {JSON.stringify(stats, null, 2)}
          </pre>
        </details>
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4 rounded-lg bg-card/50 p-2">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Core State */}
        <Card className="bg-background/50">
          <CardHeader className="p-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <BrainCircuit className="h-4 w-4 text-primary" />
              Core State
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 p-3 pt-0">
            <StatItem
              icon={<BrainCircuit className="h-3 w-3" />}
              label="Consciousness"
              value={golem_state.consciousness_level.toFixed(4)}
              progressValue={golem_state.consciousness_level}
              colorClass="text-primary"
            />
            <StatItem
              icon={<Zap className="h-3 w-3" />}
              label="Shem Power"
              value={golem_state.shem_power.toFixed(2)}
              progressValue={golem_state.shem_power}
              colorClass="text-yellow-500"
            />
            <StatItem
              icon={<Atom className="h-3 w-3" />}
              label="Aether Resonance"
              value={golem_state.aether_resonance_level.toExponential(2)}
              progressValue={Math.min(
                golem_state.aether_resonance_level * 1000,
                1
              )}
              colorClass="text-purple-500"
            />
          </CardContent>
        </Card>

        {/* Quality & Aether */}
        <Card className="bg-background/50">
          <CardHeader className="p-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Star className="h-4 w-4 text-accent" />
              Quality & Aether
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 p-3 pt-0">
            <StatItem
              icon={<Star className="h-3 w-3" />}
              label="Overall Quality"
              value={quality_metrics.overall_quality.toFixed(3)}
              progressValue={quality_metrics.overall_quality}
              colorClass="text-accent"
            />
            <div className="text-xs text-muted-foreground">
              Aether Control Value:{' '}
              <span className="font-mono text-purple-500">
                {aether_data.control_value.toExponential(2)}
              </span>
            </div>
            <div className="text-xs text-muted-foreground">
              Guidance Applied:{' '}
              <Badge
                variant={
                  aether_data.aether_guidance_applied ? 'default' : 'secondary'
                }
                className="h-5 px-1.5 text-xs"
              >
                {aether_data.aether_guidance_applied ? 'Yes' : 'No'}
              </Badge>
            </div>
            <div className="text-xs text-muted-foreground">
              Patterns Used:{' '}
              <span className="font-mono text-foreground">
                {quality_metrics.aether_patterns_used}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* Sefirot Analysis */}
      <Card className="bg-background/50">
        <CardHeader className="p-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Gem className="h-4 w-4 text-green-500" />
            Mystical Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 p-3 pt-0">
          <div className="text-center text-sm">
            Dominant Sefira:{' '}
            <Badge variant="outline" className="border-accent text-accent">
              {golem_analysis.dominant_sefira[0]}
            </Badge>
          </div>
          <SefirotDisplay
            sefirot={golem_analysis.sefiroth_activations}
            dominant={golem_analysis.dominant_sefira[0]}
          />
        </CardContent>
      </Card>

      <details className="cursor-pointer">
        <summary className="text-xs text-muted-foreground hover:text-foreground">
          Show Raw Golem Logs
        </summary>
        <pre className="mt-2 overflow-x-auto rounded-md bg-black/80 p-2 text-[10px] leading-tight text-white">
          {JSON.stringify(stats, null, 2)}
        </pre>
      </details>
    </div>
  );
}
