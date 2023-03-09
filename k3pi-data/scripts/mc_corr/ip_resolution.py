"""
Skeleton placeholder script for the IP reweighting that
we'll need to do for particle gun

"""


def main():
    """
    From
    https://indico.cern.ch/event/869306/contributions/3713441/attachments/1973743/3284152/D0mixingWS_CharmWG_22_01_20.pdf

    Could apply IP smearing:
        - For the PV:
            - get 2d histograms of (x resolution, z resolution) from data (in bins of decay time)
                - cumulative histogram st first bin is 0, each bin represents cumulative prob of being in the bin
            - randomly choose a (x res, z res) value from this histogram
                - gen random number, see which bin it corresponds to
            - Use the selected resolutions to generate a random number according to the 3d gaussian (x res, x res, z res)
            - this is the new position of the PV
            - re calculate all the variables affected by the PV moving
                - not sure what these will be
        - Also need to calculate a similar smearing for the D vertex - though not sure how to do that yet

    """


if __name__ == "__main__":
    main()
