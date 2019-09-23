//                       MFEM Example 6 - Parallel Version
//
// Compile with: make ex6p
//
// Sample runs:  mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 1
//               mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/star.mesh -o 3
//               mpirun -np 4 ex6p -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/fichera.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/ball-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/pipe-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/amr-quad.mesh
//
// Device sample runs:
//               mpirun -np 4 ex6p -pa -d cuda
//               mpirun -np 4 ex6p -pa -d occa-cuda
//               mpirun -np 4 ex6p -pa -d raja-omp
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Laplace
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <engpar.h>
#include <engpar_support.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <chrono>
#include <thread>

using namespace std;
using namespace mfem;

void switchToOriginals(int split_factor, bool& isOriginal, MPI_Comm& newComm) {
  int self = PCU_Comm_Self();
  int group;
  int groupRank;
  isOriginal = self%split_factor==0;

  if (isOriginal) {
    group=0;
    groupRank=self/split_factor;
  }
  else {
    group = 1;
    groupRank = 0;
  }
  MPI_Comm_split(MPI_COMM_WORLD,group,groupRank,&newComm);
}

int* getPartition(Mesh& m, MPI_Comm comm, int rank, int splitFactor,
    float tgtImb) {
   bool isOriginal = false;
   MPI_Comm newComm;
   switchToOriginals(splitFactor,isOriginal,newComm);
   EnGPar_Switch_Comm(newComm);
   agi::Ngraph* graph = agi::createEmptyGraph();
   long local_elems = 0;
   int* ptnVec = NULL;
   if(isOriginal) {
     auto ncm = m.ncmesh;
     ncm->PrintStats();

     local_elems = m.GetNE();
     ptnVec = new int[local_elems];

     const int rank = 0;

     std::vector<agi::gid_t> gVerts(local_elems);
     for (int i = 0; i < local_elems; i++) {
       gVerts[i] = i;
     }
     std::vector<agi::wgt_t> ignored;
     graph->constructVerts(true,gVerts,ignored);

     const auto numMeshVerts = ncm->GetNVertices();
     std::vector<agi::gid_t> gEdges(numMeshVerts);
     std::vector<agi::lid_t> gEdgeDegrees(numMeshVerts);

     auto vtxToElm = m.GetVertexToElementTable();
     const auto numPins = vtxToElm->Size_of_connections();
     std::vector<agi::gid_t> gEdgePins;
     gEdgePins.reserve(numPins);
     for(int i = 0; i< vtxToElm->Size(); i++) {
        const auto globalVtxId = i;
        gEdges[i] = globalVtxId;
        const auto deg = vtxToElm->RowSize(i);
        gEdgeDegrees[i] = deg;
     }

     std::unordered_map<agi::gid_t,agi::part_t> ghost_owners;

     auto pincount = 0;
     for(int i = 0; i < vtxToElm->Size(); i++) {
        mfem::Array<int> pins;
        vtxToElm->GetRow(i,pins);
        for(int j=0; j<pins.Size(); j++) {
          gEdgePins.push_back(pins[j]);
          pincount++;
        }
     }

     assert(pincount == gEdgePins.size());

     graph->constructEdges(gEdges,gEdgeDegrees,gEdgePins,ignored);
     graph->constructGhosts(ghost_owners);
     agi::checkValidity(graph);
   }

   agi::etype t = 0;
   engpar::Input* input_s =
     engpar::createGlobalSplitInput(graph,newComm,MPI_COMM_WORLD,
                                     isOriginal,tgtImb,t);

   //Perform split
   if(isOriginal) {
     printf("\nBefore Split\n");
     engpar::evaluatePartition(graph);
   }
   engpar::split(input_s,engpar::GLOBAL_PARMETIS);
   if(!rank)
     printf("\nAfter Split\n");
   engpar::evaluatePartition(graph);

   agi::PartitionMap* map = graph->getPartition();
   for (int i=0; i<local_elems; i++) {
     int new_owner = map->at(i);
     ptnVec[i] = new_owner;
   }
   agi::destroyGraph(graph);

   //mfem wants all ranks to have the partition vector
   MPI_Bcast(&local_elems, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(rank) ptnVec = new int[local_elems];
   MPI_Bcast(ptnVec, local_elems, MPI_INT, 0, MPI_COMM_WORLD);

   return ptnVec;
}

void writeIntElementField(const char* ptn_file, const int size, int* field) {
  ofstream out(ptn_file, ios_base::out);
  out << size << "\n";
  for(int i=0; i<size; i++)
    out << field[i] << "\n";
  out.close();
}

int* readIntElementField(const char* ptn_file, const int expectedSz) {
  ifstream in(ptn_file, ios_base::in);
  vector<string> text;
  string line;
  std::stringstream buffer;
  buffer << in.rdbuf();
  in.close();
  getline(buffer, line);
  auto size = std::stoi(line);
  if( expectedSz != size ) {
    fprintf(stderr, "expected size %d != size %d ... exiting\n",
            expectedSz, size);
    exit(EXIT_FAILURE);
  }
  int* field = new int[size];
  for(int i=0; i<size; i++) {
    getline(buffer, line);
    field[i] = std::stoi(line);
  }
  return field;
}

void getCut(ParMesh& m, int rank) {
  auto ng = m.GetNGroups();
  auto g = m.gtopo;
  assert( ng == g.NGroups() );
  auto nb = g.GetNumNeighbors();
  long cutEnts[3] = {0,0,0};
  for(int i=1; i<ng; i++) {
    if( g.IAmMaster(i) ) {
      cutEnts[0] += m.GroupNVertices(i);
      cutEnts[1] += m.GroupNEdges(i);
      cutEnts[2] += m.GroupNTriangles(i)+
                   m.GroupNQuadrilaterals(i);
    }
  }
  long totCutEnts[3] = {0,0,0};
  MPI_Allreduce(cutEnts,&totCutEnts,3,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  if(!rank) {
    printf("cut entities <vtx edge face> %8ld %8ld %8ld\n",
        totCutEnts[0], totCutEnts[1], totCutEnts[2]);
  }
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   EnGPar_Initialize();

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();
   std::this_thread::sleep_for(std::chrono::milliseconds(10));
   tic_toc.Stop();
   if( !myid )
     cout << " 10ms measured as " << tic_toc.RealTime() << " seconds\n";

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   const char *omesh_file = "";
   const char *vtkomesh_file = "";
   const char *rPtn_file = "";
   const char *wPtn_file = "";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool engpar = false;
   int metisMethod = 0;
   double tgtImb = 1.05;
   int maxiter = 5;
   int refiter = 0;
   int gpusPerNode = 1;
   bool reorder = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&omesh_file, "-om", "--outmesh",
                  "Output mesh file name.");
   args.AddOption(&vtkomesh_file, "-vtk", "--vtkoutmesh",
                  "VTK output mesh file name.");
   args.AddOption(&engpar, "-eng", "--engpar", "-no-eng",
                  "--no-engpar",
                  "Enable or disable EnGPar partitioning.");
   args.AddOption(&reorder, "-re", "--reorder",
                  "-no-re", "--no-reorder",
                  "Enable Gecko reordering.");
   args.AddOption(&metisMethod, "-mm", "--metisMethod",
                  "METIS partitioning method = [ 1 PartKWay | 2 PartVKway].");
   args.AddOption(&rPtn_file, "-rpv", "--readPtnVector",
                  "Specify file containing the partition vector.");
   args.AddOption(&wPtn_file, "-wpv", "--writePtnVector",
                  "Specify file for writing partition vector.");
   args.AddOption(&tgtImb, "-t", "--targetImbalance",
                  "Target partition imbalance; 1.05 requests a 5\% imbalance.");
   args.AddOption(&maxiter, "-i", "--iterations",
                  "Maximum AMR iterations.");
   args.AddOption(&refiter, "-r", "--refineiterations",
                  "Uniform refinement iterations.");
   args.AddOption(&gpusPerNode, "-gpn", "--gpusPerNode",
                  "Number of GPUs per node.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config, myid%gpusPerNode);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();
   std::string meshType = mesh->Nonconforming() ? "non-conforming" : "conforming";
   if( !myid )
     fprintf(stderr, "serial %s mesh vtx elms %d %d\n",
       meshType.c_str(), mesh->GetNV(), mesh->GetNE());

   // 5. Refine the serial mesh on all processors to increase the resolution.
   //    Also project a NURBS mesh to a piecewise-quadratic curved mesh. Make
   //    sure that the mesh is non-conforming.
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      mesh->SetCurvature(2);
   }

   if( reorder ) {
     if( !myid ) printf("reordering with Gecko\n");
     Array<int> ordering;
     mesh->GetGeckoElementReordering(ordering);
     mesh->ReorderElements(ordering);
   }
   mesh->EnsureNCMesh();

   for(int i=0; i<refiter; i++) {
     mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by partitioning the serial mesh.
   //    Once the parallel mesh is defined, the serial mesh can be deleted.
   tic_toc.Clear();
   tic_toc.Start();
   int* partition_vector = NULL;
   if( engpar ) {
     partition_vector = getPartition(*mesh,MPI_COMM_WORLD,myid,num_procs,tgtImb);
   } else if( metisMethod ) {
     partition_vector = mesh->GeneratePartitioning(num_procs,metisMethod,tgtImb);
   } else if( std::string(rPtn_file) != "" ) {
     if(!myid)
       printf("reading partition from \'%s\'\n",rPtn_file);
     partition_vector = readIntElementField(rPtn_file,mesh->GetNE());
   }
   tic_toc.Stop();
   auto t = tic_toc.RealTime();
   if(!myid)
     printf("partitioning time (seconds) %f\n",t);

   if( std::string(wPtn_file) != "" && !myid )
     writeIntElementField(wPtn_file,mesh->GetNE(),partition_vector);

   tic_toc.Clear();
   tic_toc.Start();
   ParMesh pmesh(MPI_COMM_WORLD, *mesh, partition_vector);
   delete mesh;
   tic_toc.Stop();
   t = tic_toc.RealTime();
   if(!myid)
     printf("ParMesh created in (seconds) %f\n",t);

   {
      ostringstream mesh_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);
   }

   {
     const int order = 1;
     const int dim = pmesh.Dimension();
     FiniteElementCollection* fec = new H1_FECollection(order, dim);
     ParFiniteElementSpace *fespace =
       new ParFiniteElementSpace(&pmesh, fec, dim, Ordering::byVDIM);

     auto pncm = pmesh.pncmesh;
     const auto nlverts = pncm->GetNVertices();
     const auto ngverts = pncm->GetNGhostVertices();
     const auto nlelms = pncm->GetNElements();
     const auto ngelms = pncm->GetNGhostElements();
     printf("%d numLocalElms %d numGhostsElms %d numLocalVerts %d numGhostVerts %d\n",
       myid, nlelms, ngelms, nlverts, ngverts);
     for(int i=0; i<ngelms; i++) {
        auto ghostGid = pmesh.pncmesh->GetLeafGlobId(i+nlelms);
        auto ghostRank = pmesh.pncmesh->ElementRank(i);
        printf("%d ghostLid ghostGid ghostRank %5d %5ld %5d\n", myid, i, ghostGid, ghostRank);
     }
     Array<int> vtxIds;
     auto vtxToElm = pmesh.pncmesh->GetVertexToElementTable(myid,vtxIds);
     if(!myid) vtxToElm->Print();
     auto m_nverts = vtxToElm->Size();
     std::vector<agi::gid_t> gEdges(m_nverts);
     std::vector<agi::lid_t> gEdgeDegrees(m_nverts);
     for(int i = 0; i< vtxToElm->Size(); i++) {
        gEdges[i] = fespace->GetGlobalTDofNumber(vtxIds[i]);
        gEdgeDegrees[i] = vtxToElm->RowSize(i);
        printf("%3d globvtx %3ld deg %3d\n", myid, gEdges[i], gEdgeDegrees[i]);
     }

     std::vector<agi::gid_t> gVerts(nlelms);
     for (int i = 0; i < nlelms; i++) {
       gVerts[i] = i;
     }

     std::vector<agi::wgt_t> ignored;
     agi::Ngraph* graph = agi::createEmptyGraph();
     graph->constructVerts(true,gVerts,ignored);

     std::unordered_map<agi::gid_t,agi::part_t> ghost_owners;

     const auto numPins = vtxToElm->Size_of_connections();
     std::vector<agi::gid_t> gEdgePins;
     gEdgePins.reserve(numPins);
     auto pincount = 0;
     for(int i = 0; i < vtxToElm->Size(); i++) {
        mfem::Array<int> pins;
        vtxToElm->GetRow(i,pins);
        for(int j=0; j<pins.Size(); j++) {
          const auto elm = pins[j];
          const auto elmRank = pmesh.pncmesh->ElementRank(elm);
          const auto globElmId = pmesh.pncmesh->GetLeafGlobId(elm);
          gEdgePins.push_back(globElmId);
          pincount++;
          if( elm > nlelms ) {
            ghost_owners[globElmId] = elmRank;
            assert(elmRank!=myid);
          }
        }
     }

     assert(pincount == gEdgePins.size());
     graph->constructEdges(gEdges,gEdgeDegrees,gEdgePins,ignored);
     graph->constructGhosts(ghost_owners);
     agi::checkValidity(graph);

     MPI_Barrier(MPI_COMM_WORLD);
     printf("%d done\n", myid);
     delete fespace;
     delete fec;
   }

   return 0;

   getCut(pmesh,myid);

   MFEM_VERIFY(pmesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 7. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   // 8. As in Example 1p, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   ParLinearForm b(&fespace);

   ConstantCoefficient one(1.0);

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // 9. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   ParGridFunction x(&fespace);
   x = 0;

   // 10. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sout;
   if (visualization)
   {
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
         visualization = false;
      }

      sout.precision(8);
   }

   // 11. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(&pmesh, &flux_fec, sdim);
   RT_FECollection smooth_flux_fec(order-1, dim);
   ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);
   L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fes, smooth_flux_fes);

   // 12. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   // 13. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 10000000;
   for (int it = 0; it < maxiter; it++)
   {
      HYPRE_Int global_dofs = fespace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      // 14. Assemble the right-hand side and determine the list of true
      //     (i.e. parallel conforming) essential boundary dofs.
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      b.Assemble();

      pmesh.PrintInfo();
      // 15. Assemble the stiffness matrix. Note that MFEM doesn't care at this
      //     point that the mesh is nonconforming and parallel.  The FE space is
      //     considered 'cut' along hanging edges/faces, and also across
      //     processor boundaries.
      //MPI_Pcontrol(1); //start mpip profiling
      tic_toc.Clear();
      tic_toc.Start();
      a.Assemble();

      // 16. Create the parallel linear system: eliminate boundary conditions.
      //     The system will be solved for true (unconstrained/unique) DOFs only.
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

      // 17. Solve the linear system A X = B.
      //     * With full assembly, use the BoomerAMG preconditioner from hypre.
      //     * With partial assembly, use no preconditioner, for now.
      HypreBoomerAMG *amg = NULL;
      if (!pa) { amg = new HypreBoomerAMG; amg->SetPrintLevel(0); }
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-6);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(3); // print the first and the last iterations only
      if (amg) { cg.SetPreconditioner(*amg); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete amg;

      tic_toc.Stop();

      //MPI_Pcontrol(0); //stop mpip profiling
      {
        double t = tic_toc.RealTime();
        double maxt;
        MPI_Allreduce(&t,&maxt,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if(!myid)
          printf("Max elapsed GPU work time (seconds) %f\n",maxt);
        double* allT = NULL;
        if(!myid)
          allT = new double[num_procs];
        MPI_Gather(&t,1,MPI_DOUBLE,allT,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(!myid) {
          fprintf(stderr,"<rank> <GPU work time>\n");
          for(int i=0; i<num_procs; i++)
            fprintf(stderr,"%4d %6f\n", i, allT[i]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      // 18. Switch back to the host and extract the parallel grid function
      //     corresponding to the finite element approximation X. This is the
      //     local solution on each processor.
      a.RecoverFEMSolution(X, b, x);

      // 19. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout << "solution\n" << pmesh << x << flush;
      }

      if (global_dofs > max_dofs)
      {
         if (myid == 0)
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (myid == 0)
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 21. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      fespace.Update();
      x.Update();

      // 22. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         fespace.Update();
         x.Update();
      }

      // 23. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }

   std::string omesh_name(omesh_file);
   if( omesh_name != "" ) {
     ofgzstream omesh_stream(omesh_name.c_str(), "w");
     pmesh.ParPrint(omesh_stream);
   }
   std::string vtkomesh_name(vtkomesh_file);
   if( vtkomesh_name != "" ) {
     std::stringstream ss;
     ss << vtkomesh_name << myid << ".vtk";
     std::string s = ss.str();
     printf("writing vtk mesh as %s\n", s.c_str());
     ofstream ostream(s.c_str());
     ostream.precision(14);
     pmesh.PrintVTK(ostream,0,1);
   }

   EnGPar_Finalize();
   MPI_Finalize();
   return 0;
}
