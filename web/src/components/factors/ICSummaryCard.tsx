import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Tooltip from '@mui/material/Tooltip';

interface ICValues {
  icMean: number;
  icir: number;
  rankIcMean: number;
  rankIcir: number;
}

interface ICSummaryCardProps {
  data: ICValues;
  factorName?: string;
}

function ICStatBox({ label, value, tooltip }: { label: string; value: string; tooltip: string }) {
  return (
    <Tooltip title={tooltip}>
      <Box sx={{ textAlign: 'center', p: 1 }}>
        <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase' }}>
          {label}
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {value}
        </Typography>
      </Box>
    </Tooltip>
  );
}

export function ICSummaryCard({ data, factorName }: ICSummaryCardProps) {
  return (
    <Card>
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          {factorName ? `${factorName} IC Summary` : 'IC Summary'}
        </Typography>
        <Grid container>
          <Grid item xs={3}>
            <ICStatBox
              label="IC Mean"
              value={data.icMean.toFixed(4)}
              tooltip="Average Information Coefficient across periods"
            />
          </Grid>
          <Grid item xs={3}>
            <ICStatBox
              label="ICIR"
              value={data.icir.toFixed(4)}
              tooltip="IC / Std(IC) — Information Coefficient Information Ratio"
            />
          </Grid>
          <Grid item xs={3}>
            <ICStatBox
              label="Rank IC"
              value={data.rankIcMean.toFixed(4)}
              tooltip="Spearman Rank IC — correlation of rank order"
            />
          </Grid>
          <Grid item xs={3}>
            <ICStatBox
              label="Rank ICIR"
              value={data.rankIcir.toFixed(4)}
              tooltip="Rank IC / Std(Rank IC)"
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

/** Generate mock IC summary data */
export function generateMockICSummary(): ICValues {
  let rng = 123;
  const nextRand = () => {
    rng = (rng * 16807 + 0) % 2147483647;
    return (rng - 1) / 2147483646;
  };

  const icMean = 0.03 + nextRand() * 0.04;
  return {
    icMean: Math.round(icMean * 10000) / 10000,
    icir: Math.round((icMean / (0.05 + nextRand() * 0.03)) * 10000) / 10000,
    rankIcMean: Math.round((0.02 + nextRand() * 0.04) * 10000) / 10000,
    rankIcir: Math.round((0.5 + nextRand() * 1.0) * 10000) / 10000,
  };
}
